"""
Analytics pipeline configuration.

This defines the entire flow:
  Client → Staging → Central Database

Both export (client) and ingestion (central) use this same config.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class S3Config:
    """
    S3 storage configuration.

    Used for:
    - Staging area (parquet file export/ingest)
    - DuckLake data path (ducklake storage backend)
    """

    key: str = ""  # Access key ID
    secret: str = ""  # Secret access key
    region: str = "us-east-1"
    endpoint: str = ""  # e.g., "https://s3.amazonaws.com" or "http://localhost:9000"
    url_style: Literal["path", "vhost"] = "path"  # "vhost" = bucket.endpoint/key

    def __str__(self):
        # Mask credentials for display
        masked_key = f"{self.key[:4]}...{self.key[-4:]}" if len(self.key) > 8 else "***"
        return f"S3({self.endpoint}, key={masked_key})"


@dataclass
class PostgresConfig:
    """Postgres connection configuration (for DuckLake metadata)"""

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = ""
    dbname: str = "ducklake"

    def connection_string(self) -> str:
        """Get postgres connection string for DuckLake ATTACH"""
        return f"postgres:dbname={self.dbname} user={self.user} password={self.password} host={self.host} port={self.port}"


@dataclass
class StagingS3:
    """Stage parquet files in S3 bucket"""

    type: Literal["s3"] = "s3"
    s3: S3Config = field(default_factory=S3Config)
    bucket: str = ""
    prefix: str = "staging"
    archive_prefix: str = "archive"


@dataclass
class StagingFilesystem:
    """Stage parquet files on local/shared filesystem"""

    type: Literal["filesystem"] = "filesystem"
    staging_dir: Path = field(default_factory=lambda: Path("./staging"))
    archive_dir: Path = field(default_factory=lambda: Path("./archive"))


@dataclass
class FeedTarget:
    """Publish local analytics to a project-scoped Feed endpoint."""

    type: Literal["feed"] = "feed"
    project: Optional[str] = None
    server_url: Optional[str] = None
    chunk_size: int = 1000
    enqueue_timeout_seconds: float = 30.0
    flush_timeout_seconds: float = 30.0
    spool_dir: Optional[str] = None
    feed: Optional[str] = None

    def __post_init__(self):
        if self.feed and self.project and self.feed != self.project:
            raise ValueError("FeedTarget.feed and legacy project must identify the same feed")

    @property
    def destination(self) -> Optional[str]:
        return self.feed or self.project


@dataclass
class CentralDuckDB:
    """Central database as local DuckDB file"""

    type: Literal["duckdb"] = "duckdb"
    db_path: Path = field(default_factory=lambda: Path("./central.db"))


@dataclass
class CentralDuckLake:
    """
    Central database as remote DuckLake.

    DuckLake requires:
    - Postgres: Metadata database
    - S3: Data storage backend
    """

    type: Literal["ducklake"] = "ducklake"
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    s3: S3Config = field(default_factory=S3Config)
    data_path: str = "s3://ducklake-data"  # S3 path for ducklake storage


@dataclass
class AnalyticsConfig:
    """
    Complete analytics pipeline configuration.

    Defines:
    - Where clients export parquet files (staging)
    - Where ingestion reads from (same staging)
    - Where ingestion writes to (central database)

    Examples:

    # Local development (filesystem staging, local DB)
    AnalyticsConfig(
        staging=StagingFilesystem(),
        central=CentralDuckDB()
    )

    # Production (S3 staging, DuckLake central) - shared S3 config
    s3 = S3Config(
        key="...",
        secret="...",
        endpoint="https://s3.amazonaws.com"
    )
    AnalyticsConfig(
        staging=StagingS3(s3=s3, bucket="metrics-staging"),
        central=CentralDuckLake(
            postgres=PostgresConfig(host="db.prod.com", password="..."),
            s3=s3,
            data_path="s3://ducklake-data"
        )
    )

    # Hybrid (S3 staging, local DB - for testing S3 without DuckLake)
    AnalyticsConfig(
        staging=StagingS3(
            s3=S3Config(endpoint="http://localhost:9000", key="minioadmin", secret="minioadmin"),
            bucket="test"
        ),
        central=CentralDuckDB(db_path=Path("test.db"))
    )
    """

    staging: StagingS3 | StagingFilesystem | FeedTarget = field(
        default_factory=lambda: StagingFilesystem()
    )
    central: CentralDuckDB | CentralDuckLake = field(
        default_factory=lambda: CentralDuckDB()
    )

    # Export/ingest interval settings
    export_interval_seconds: int = 300  # How often clients export
    ingest_interval_seconds: int = 300  # How often central ingests

    # Optional healthchecks.io ping URL for continuous ingest monitoring
    healthcheck_url: str = ""  # e.g. "https://hc-ping.com/your-uuid-here"

    def is_s3_staging(self) -> bool:
        """Check if using S3 for staging"""
        return self.staging.type == "s3"

    def is_filesystem_staging(self) -> bool:
        """Check if using filesystem for staging"""
        return self.staging.type == "filesystem"

    def is_feed(self) -> bool:
        """Check if clients publish through Feed instead of Parquet staging."""
        return self.staging.type == "feed"

    def is_ducklake_central(self) -> bool:
        """Check if central is DuckLake"""
        return self.central.type == "ducklake"

    def is_duckdb_central(self) -> bool:
        """Check if central is local DuckDB"""
        return self.central.type == "duckdb"

    def __str__(self):
        if self.is_s3_staging():
            staging_desc = f"S3 ({self.staging.bucket})"
        elif self.is_filesystem_staging():
            staging_desc = f"Filesystem ({self.staging.staging_dir})"
        else:
            staging_desc = f"Feed ({self.staging.project})"
        if self.is_feed():
            return f"Analytics: {staging_desc}"
        central_desc = (
            f"DuckLake ({self.central.postgres.host})"
            if self.is_ducklake_central()
            else f"DuckDB ({self.central.db_path})"
        )
        return f"Analytics: {staging_desc} → {central_desc}"


# Global config - can be overridden by importing module
_analytics_config: Optional[AnalyticsConfig] = None


def analytics_config() -> AnalyticsConfig:
    """
    Get the current analytics configuration.

    Loads from env.py if available, otherwise uses defaults.
    """
    global _analytics_config

    if _analytics_config is not None:
        return _analytics_config

    # Try to load from env.py
    try:
        from env import get_analytics_config

        _analytics_config = get_analytics_config()
        print(f"[analytics] Loaded config from env.py: {_analytics_config}")
        return _analytics_config
    except ImportError:
        pass
    except AttributeError:
        pass

    # Default configuration (local filesystem + local DB)
    _analytics_config = AnalyticsConfig()
    print(f"[analytics] Using default config: {_analytics_config}")
    return _analytics_config


def set_analytics_config(config: AnalyticsConfig):
    """Override the global analytics configuration"""
    global _analytics_config
    _analytics_config = config
