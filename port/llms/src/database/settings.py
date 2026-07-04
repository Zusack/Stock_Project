import sqlite3

class SettingsRepository:
    def get_setting(self, key: str, default: str = None) -> str:
        if not self.connection: raise sqlite3.Error("DB not open")
        cursor = self.connection.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row['value'] if row else default

    def set_setting(self, key: str, value: str):
        if not self.connection: raise sqlite3.Error("DB not open")
        self.connection.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        self.connection.commit()