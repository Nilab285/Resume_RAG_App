import sqlite3
import time
from contextlib import contextmanager
from config import DB_PATH, SQLITE_BUSY_TIMEOUT

MAX_DB_RETRIES = 5
RETRY_DELAY = 0.2  # seconds


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Returns a SQLite connection configured for concurrent access.
    """

    conn = sqlite3.connect(
        db_path,
        timeout=SQLITE_BUSY_TIMEOUT / 1000,
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    # Better concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT};")
    conn.execute("PRAGMA foreign_keys=ON;")

    return conn


@contextmanager
def transaction(db_path: str = DB_PATH):
    """
    Automatically commits or rolls back a transaction.
    """

    conn = get_connection(db_path)

    try:
        yield conn
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


def execute(query, params=(), db_path: str = DB_PATH):

    for attempt in range(MAX_DB_RETRIES):

        try:
            with transaction(db_path) as conn:
                conn.execute(query, params)
            return

        except sqlite3.OperationalError as e:

            if "locked" not in str(e).lower():
                raise

            if attempt == MAX_DB_RETRIES - 1:
                raise

            time.sleep(RETRY_DELAY * (attempt + 1))


def executemany(query, rows, db_path: str = DB_PATH):

    for attempt in range(MAX_DB_RETRIES):

        try:

            with transaction(db_path) as conn:
                conn.executemany(query, rows)

            return

        except sqlite3.OperationalError as e:

            if "locked" not in str(e).lower():
                raise

            if attempt == MAX_DB_RETRIES - 1:
                raise

            time.sleep(RETRY_DELAY * (attempt + 1))


def fetchone(query, params=(), db_path: str = DB_PATH):

    with get_connection(db_path) as conn:

        cur = conn.execute(query, params)

        row = cur.fetchone()

        return dict(row) if row else None


def fetchall(query, params=(), db_path: str = DB_PATH):

    with get_connection(db_path) as conn:

        cur = conn.execute(query, params)

        return [dict(r) for r in cur.fetchall()]