import sqlite3
import os
import time

class DatabaseCore:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.connection = None
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.db_file)), exist_ok=True)

    def connect(self) -> None:
        try:
            self.connection = sqlite3.connect(self.db_file, detect_types=sqlite3.PARSE_DECLTYPES)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON;")
            
            # --- FIX: Safe WAL Initialization ---
            # Tolerate "disk I/O error" on Windows if file is momentarily busy
            try:
                self.connection.execute("PRAGMA journal_mode=WAL;") 
                self.connection.execute("PRAGMA synchronous=NORMAL;")
            except sqlite3.OperationalError as e:
                if "disk I/O error" in str(e):
                    print(f"Warning: Could not set WAL mode (Disk I/O). Continuing. Error: {e}")
                else:
                    raise

        except sqlite3.Error as e:
            print(f"Error connecting to database {self.db_file}: {e}")
            raise

    def close(self) -> None:
        if self.connection:
            try:
                self.connection.commit()
                self.connection.close()
            except sqlite3.Error as e:
                print(f"Error closing database: {e}")
            finally:
                self.connection = None

    def __enter__(self):
        # Retry logic for connection
        retries = 3
        for i in range(retries):
            try:
                self.connect()
                return self
            except sqlite3.OperationalError as e:
                if "disk I/O error" in str(e) or "database is locked" in str(e):
                    if i < retries - 1:
                        time.sleep(0.2)
                        continue
                raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def reset_database(self) -> bool:
        self.close()
        if os.path.exists(self.db_file):
            try:
                os.remove(self.db_file)
                return True
            except OSError as e:
                print(f"Error removing database file: {e}")
                return False
        return True