"""PostgreSQL schema (SQLAlchemy 2.0). Also runs on SQLite, which the tests use."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# SQLite only autoincrements INTEGER primary keys, so BigInteger falls back to Integer there.
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(dt: datetime) -> datetime:
    """SQLite hands back naive datetimes even for timezone=True columns; Postgres doesn't."""
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    pass


class PairStatus:
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class MarketPair(Base):
    """A Polymarket market and a Kalshi market believed to resolve on the same proposition."""

    __tablename__ = "market_pairs"
    __table_args__ = (UniqueConstraint("poly_market_id", "kalshi_ticker", name="uq_pair"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    poly_market_id: Mapped[str] = mapped_column(String(64), index=True)
    poly_question: Mapped[str] = mapped_column(Text)
    poly_event_title: Mapped[str] = mapped_column(Text, default="")
    poly_yes_outcome: Mapped[str] = mapped_column(String(200), default="Yes")
    poly_slug: Mapped[str] = mapped_column(String(300), default="")
    poly_yes_token: Mapped[str] = mapped_column(String(100))
    poly_no_token: Mapped[str] = mapped_column(String(100))
    poly_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    poly_fee_rate: Mapped[float] = mapped_column(Float, default=0.0)
    poly_fee_exponent: Mapped[float] = mapped_column(Float, default=1.0)
    poly_rules: Mapped[str] = mapped_column(Text, default="")

    kalshi_ticker: Mapped[str] = mapped_column(String(120), index=True)
    kalshi_series: Mapped[str] = mapped_column(String(120), default="")
    kalshi_title: Mapped[str] = mapped_column(Text)
    kalshi_resolution: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    kalshi_fee_multiplier: Mapped[float | None] = mapped_column(Float)
    kalshi_category: Mapped[str] = mapped_column(String(80), default="")
    kalshi_rules: Mapped[str] = mapped_column(Text, default="")

    text_similarity: Mapped[float] = mapped_column(Float)
    resolution_confidence: Mapped[float] = mapped_column(Float, index=True)
    resolution_gap_days: Mapped[float | None] = mapped_column(Float)
    flags: Mapped[list] = mapped_column(JSON, default=list)
    suggest_inverted: Mapped[bool] = mapped_column(Boolean, default=False)
    inverted: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default=PairStatus.CANDIDATE, index=True)
    review_note: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    # Last time both markets were present in a fetched universe; stale pairs are not scanned.
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)

    @property
    def resolution_time(self) -> datetime | None:
        """Capital is locked until the *later* venue resolves."""
        times = [as_utc(t) for t in (self.poly_end, self.kalshi_resolution) if t is not None]
        return max(times) if times else None


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    pairs_checked: Mapped[int] = mapped_column(Integer, default=0)
    books_walked: Mapped[int] = mapped_column(Integer, default=0)
    gross_arbs: Mapped[int] = mapped_column(Integer, default=0)
    net_arbs: Mapped[int] = mapped_column(Integer, default=0)
    observations_logged: Mapped[int] = mapped_column(Integer, default=0)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text)


class Observation(Base):
    """One pair evaluated in one scan (best direction). All returns are fractions (0.01 = 1%)."""

    __tablename__ = "observations"
    __table_args__ = (Index("ix_obs_pair_scan", "pair_id", "scan_id"),)

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True)
    scan_id: Mapped[int] = mapped_column(ForeignKey("scans.id", ondelete="CASCADE"), index=True)
    pair_id: Mapped[int] = mapped_column(ForeignKey("market_pairs.id", ondelete="CASCADE"))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)

    direction: Mapped[str] = mapped_column(String(20))
    poly_yes_ask: Mapped[float | None] = mapped_column(Float)
    poly_no_ask: Mapped[float | None] = mapped_column(Float)
    kalshi_yes_ask: Mapped[float | None] = mapped_column(Float)
    kalshi_no_ask: Mapped[float | None] = mapped_column(Float)
    poly_price: Mapped[float | None] = mapped_column(Float)  # ask paid on the Polymarket leg
    kalshi_price: Mapped[float | None] = mapped_column(Float)  # ask paid on the Kalshi leg

    gross_cost_top: Mapped[float | None] = mapped_column(Float)
    gross_return_top: Mapped[float | None] = mapped_column(Float)
    net_return_top: Mapped[float | None] = mapped_column(Float)
    book_walked: Mapped[bool] = mapped_column(Boolean, default=False)

    max_size: Mapped[int] = mapped_column(Integer, default=0)  # profit-maximizing contracts
    capital: Mapped[float] = mapped_column(Float, default=0.0)
    profit: Mapped[float] = mapped_column(Float, default=0.0)
    net_return: Mapped[float | None] = mapped_column(Float)  # fee-adjusted, at max_size
    annualized_return: Mapped[float | None] = mapped_column(Float)
    excess_annualized_return: Mapped[float | None] = mapped_column(Float)
    days_to_resolution: Mapped[float | None] = mapped_column(Float)

    resolution_confidence: Mapped[float] = mapped_column(Float)
    pair_status: Mapped[str] = mapped_column(String(16))
    is_net_arb: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    profit_curve: Mapped[list | None] = mapped_column(JSON)


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True, future=True)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(engine, expire_on_commit=False, future=True)
