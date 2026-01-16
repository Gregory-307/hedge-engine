# SQLAlchemy engine for decision audit logging
from sqlalchemy import create_engine
from .config import Settings

settings = Settings()
engine = create_engine(settings.db_dsn, future=True)
