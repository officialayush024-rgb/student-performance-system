
import sqlite3
import os

os.makedirs("data", exist_ok=True)

DB_PATH = "data/students.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_code TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                class_name TEXT NOT NULL,
                attendance_pct REAL NOT NULL,
                study_hours_per_day REAL NOT NULL,
                previous_score REAL NOT NULL,
                assignment_score REAL NOT NULL,
                sleep_hours REAL NOT NULL,
                participation_level INTEGER NOT NULL,
                actual_final_score REAL,
                predicted_score REAL,
                performance_category TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

if __name__ == "__main__":
    initialize_database()
    print("Database initialized successfully.")
