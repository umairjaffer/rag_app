"""
database.py

PostgreSQL database configuration using SQLAlchemy.

The database is used for application-level data such as:

- Sessions
- Documents
- Chat history
- Message references
- Processing logs

Qdrant remains responsible for vectors and chunk retrieval.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


# ----------------------------------------------------------------------
# Database engine
# ----------------------------------------------------------------------

engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
)


# ----------------------------------------------------------------------
# Database session factory
# ----------------------------------------------------------------------

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)


# ----------------------------------------------------------------------
# Base class for SQLAlchemy models
# ----------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ----------------------------------------------------------------------
# FastAPI database dependency
# ----------------------------------------------------------------------

def get_db():
    """
    Create one database session for the current request.

    The session is always closed after the request finishes.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()