import sqlite3
import json

DB_FILE = "judge.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                submission_json TEXT,
                evaluation_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS viva_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                persona TEXT,
                history_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(project_id) REFERENCES projects(id)
            )
        """)
        conn.commit()

def save_project(submission: dict, evaluation: dict) -> int:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO projects (title, submission_json, evaluation_json) VALUES (?, ?, ?)",
            (submission["title"], json.dumps(submission), json.dumps(evaluation))
        )
        conn.commit()
        return cursor.lastrowid

def get_project(project_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT submission_json, evaluation_json FROM projects WHERE id = ?", (project_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0]), json.loads(row[1])
        return None, None

def save_viva_session(project_id: int, persona: str, initial_history: list) -> int:
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO viva_sessions (project_id, persona, history_json) VALUES (?, ?, ?)",
            (project_id, persona, json.dumps(initial_history))
        )
        conn.commit()
        return cursor.lastrowid

def update_viva_history(session_id: int, history: list):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE viva_sessions SET history_json = ? WHERE id = ?", (json.dumps(history), session_id))
        conn.commit()

def get_viva_session(session_id: int):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT project_id, persona, history_json FROM viva_sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            return {"project_id": row[0], "persona": row[1], "history": json.loads(row[2])}
        return None