
import sqlite3
from pathlib import Path

# Always create the database inside the backend folder
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "argus_accounts.sqlite3"


def create_database():
    connection = sqlite3.connect(DB_PATH)

    try:
        connection.execute("PRAGMA foreign_keys = ON")

        connection.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS auth_sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            updated_at INTEGER NOT NULL,
            FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS
            idx_auth_sessions_user
            ON auth_sessions(user_id);

        CREATE INDEX IF NOT EXISTS
            idx_conversations_user
            ON conversations(user_id);
        """)

        connection.commit()

        print("ARGUS database created successfully!")
        print(f"Database location: {DB_PATH}")

        tables = connection.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
        """).fetchall()

        print("\nDatabase tables:")
        for table in tables:
            print("-", table[0])

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    create_database()
