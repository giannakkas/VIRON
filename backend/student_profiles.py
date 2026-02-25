"""
VIRON Student Profiles — Memory, Progress, Gamification
========================================================
SQLite database that remembers each student across sessions.
Tracks: learning progress, quiz scores, points, streaks, homework history.

Used by: face recognition (auto-load profile), quiz mode, gamification,
         homework helper, and adaptive difficulty.
"""

import sqlite3
import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viron_students.db")

# ═══════════════════════════════════════════════════════
# Points & Levels Configuration
# ═══════════════════════════════════════════════════════

POINTS = {
    "question_asked": 5,        # Ask any question
    "quiz_correct": 20,         # Correct quiz answer
    "quiz_wrong": 5,            # Wrong answer (still tried!)
    "quiz_completed": 50,       # Finish a full quiz
    "streak_bonus": 10,         # Per day of streak
    "homework_scanned": 15,     # Scanned homework for help
    "study_5min": 10,           # Every 5 minutes of study
    "first_today": 25,          # First interaction of the day
    "perfect_quiz": 100,        # 100% on a quiz
}

LEVELS = [
    (0,      "Αρχάριος",     "Beginner"),        # 0+
    (100,    "Μαθητής",      "Student"),          # 100+
    (300,    "Σπουδαστής",   "Scholar"),          # 300+
    (600,    "Ερευνητής",    "Researcher"),       # 600+
    (1000,   "Σοφός",        "Wise One"),         # 1000+
    (1500,   "Μέντορας",     "Mentor"),           # 1500+
    (2500,   "Δάσκαλος",     "Master"),           # 2500+
    (4000,   "Καθηγητής",    "Professor"),        # 4000+
    (6000,   "Διάνοια",      "Genius"),           # 6000+
    (10000,  "Θρύλος",       "Legend"),            # 10000+
]

ACHIEVEMENTS = {
    "first_question":   {"name_el": "Πρώτη Ερώτηση",   "name_en": "First Question",    "desc": "Asked your first question",          "icon": "🎯"},
    "streak_3":         {"name_el": "3 Μέρες Σερί",    "name_en": "3-Day Streak",       "desc": "Studied 3 days in a row",             "icon": "🔥"},
    "streak_7":         {"name_el": "Εβδομάδα Φωτιά",  "name_en": "Week on Fire",       "desc": "Studied 7 days in a row",             "icon": "⚡"},
    "streak_30":        {"name_el": "Μήνας Αφοσίωσης", "name_en": "Month of Devotion",  "desc": "Studied 30 days in a row",            "icon": "👑"},
    "quiz_master":      {"name_el": "Quiz Master",      "name_en": "Quiz Master",        "desc": "Got 100% on 5 quizzes",               "icon": "🏆"},
    "math_whiz":        {"name_el": "Μαθηματικός",      "name_en": "Math Whiz",          "desc": "Answered 50 math questions correctly", "icon": "🔢"},
    "science_explorer": {"name_el": "Εξερευνητής",      "name_en": "Science Explorer",   "desc": "Asked 30 science questions",           "icon": "🔬"},
    "bookworm":         {"name_el": "Βιβλιοφάγος",      "name_en": "Bookworm",           "desc": "Studied for 10 hours total",           "icon": "📚"},
    "night_owl":        {"name_el": "Νυχτοπούλι",       "name_en": "Night Owl",          "desc": "Studied after 9 PM",                   "icon": "🦉"},
    "early_bird":       {"name_el": "Πρωινό Πουλί",    "name_en": "Early Bird",          "desc": "Studied before 8 AM",                  "icon": "🐦"},
    "homework_hero":    {"name_el": "Ήρωας Εργασιών",  "name_en": "Homework Hero",      "desc": "Scanned 10 homework problems",         "icon": "📝"},
    "polyglot":         {"name_el": "Πολύγλωσσος",      "name_en": "Polyglot",           "desc": "Asked questions in 2+ languages",      "icon": "🌍"},
    "curious_mind":     {"name_el": "Περίεργο Μυαλό",  "name_en": "Curious Mind",       "desc": "Asked 100 questions total",             "icon": "💡"},
    "level_5":          {"name_el": "Σοφός",            "name_en": "Wise One",           "desc": "Reached level 5",                      "icon": "⭐"},
    "level_10":         {"name_el": "Θρύλος",           "name_en": "Legend",             "desc": "Reached level 10",                     "icon": "🌟"},
}


# ═══════════════════════════════════════════════════════
# Database Setup
# ═══════════════════════════════════════════════════════

def get_db() -> sqlite3.Connection:
    """Get a database connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_db()
    try:
        conn.executescript("""
            -- Student profiles (linked to face recognition names)
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,          -- matches face_recognizer name
                display_name TEXT,                  -- how VIRON addresses them
                grade TEXT DEFAULT '',              -- e.g. "6th", "Γ' Γυμνασίου"
                age INTEGER DEFAULT 0,
                language TEXT DEFAULT 'el',          -- preferred language
                difficulty TEXT DEFAULT 'normal',    -- easy/normal/hard
                total_points INTEGER DEFAULT 0,
                current_streak INTEGER DEFAULT 0,
                longest_streak INTEGER DEFAULT 0,
                total_study_minutes INTEGER DEFAULT 0,
                total_questions INTEGER DEFAULT 0,
                total_quizzes INTEGER DEFAULT 0,
                perfect_quizzes INTEGER DEFAULT 0,
                homework_scanned INTEGER DEFAULT 0,
                languages_used TEXT DEFAULT '[]',    -- JSON array
                achievements TEXT DEFAULT '[]',      -- JSON array of achievement IDs
                favorite_subjects TEXT DEFAULT '[]', -- JSON array
                struggling_subjects TEXT DEFAULT '[]', -- JSON array
                notes TEXT DEFAULT '',               -- teacher/parent notes
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_study_date TEXT DEFAULT ''      -- for streak tracking (YYYY-MM-DD)
            );

            -- Session log (each time a student interacts)
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                duration_minutes REAL DEFAULT 0,
                questions_asked INTEGER DEFAULT 0,
                subjects_covered TEXT DEFAULT '[]',  -- JSON array
                mood_start TEXT DEFAULT '',           -- detected emotion at start
                mood_end TEXT DEFAULT '',             -- detected emotion at end
                points_earned INTEGER DEFAULT 0,
                FOREIGN KEY (student_id) REFERENCES students(id)
            );

            -- Individual question/interaction log
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                session_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                type TEXT DEFAULT 'question',        -- question/quiz/homework/chat
                subject TEXT DEFAULT 'general',
                language TEXT DEFAULT 'el',
                question TEXT,
                answer TEXT,
                was_correct INTEGER DEFAULT -1,      -- -1=n/a, 0=wrong, 1=correct
                difficulty TEXT DEFAULT 'normal',
                points_earned INTEGER DEFAULT 0,
                emotion TEXT DEFAULT '',
                FOREIGN KEY (student_id) REFERENCES students(id),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            -- Quiz results
            CREATE TABLE IF NOT EXISTS quizzes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                session_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                subject TEXT,
                difficulty TEXT DEFAULT 'normal',
                total_questions INTEGER DEFAULT 0,
                correct_answers INTEGER DEFAULT 0,
                score_percent REAL DEFAULT 0,
                questions_json TEXT DEFAULT '[]',     -- Full Q&A data
                time_taken_seconds INTEGER DEFAULT 0,
                points_earned INTEGER DEFAULT 0,
                FOREIGN KEY (student_id) REFERENCES students(id)
            );

            -- Homework scans
            CREATE TABLE IF NOT EXISTS homework (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                session_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                subject TEXT DEFAULT '',
                image_path TEXT DEFAULT '',
                extracted_text TEXT DEFAULT '',
                explanation TEXT DEFAULT '',
                was_helpful INTEGER DEFAULT -1,      -- student feedback
                FOREIGN KEY (student_id) REFERENCES students(id)
            );

            -- Subject progress tracking
            CREATE TABLE IF NOT EXISTS subject_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                subject TEXT NOT NULL,
                questions_asked INTEGER DEFAULT 0,
                correct_answers INTEGER DEFAULT 0,
                difficulty_level TEXT DEFAULT 'normal',
                last_topic TEXT DEFAULT '',
                mastery_score REAL DEFAULT 0.0,       -- 0.0 to 1.0
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(student_id, subject),
                FOREIGN KEY (student_id) REFERENCES students(id)
            );

            -- Create indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_sessions_student ON sessions(student_id);
            CREATE INDEX IF NOT EXISTS idx_interactions_student ON interactions(student_id);
            CREATE INDEX IF NOT EXISTS idx_interactions_type ON interactions(type);
            CREATE INDEX IF NOT EXISTS idx_quizzes_student ON quizzes(student_id);
        """)
        conn.commit()
        print("📊 Student database initialized")
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════
# Student Profile CRUD
# ═══════════════════════════════════════════════════════

def get_or_create_student(name: str) -> Dict:
    """Get student by name (from face recognition), create if new."""
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM students WHERE name = ?", (name,)).fetchone()
        if row:
            # Update last_seen
            conn.execute("UPDATE students SET last_seen = CURRENT_TIMESTAMP WHERE name = ?", (name,))
            conn.commit()
            return dict(row)
        
        # New student — create profile
        conn.execute(
            "INSERT INTO students (name, display_name) VALUES (?, ?)",
            (name, name)
        )
        conn.commit()
        row = conn.execute("SELECT * FROM students WHERE name = ?", (name,)).fetchone()
        print(f"  👤 New student profile created: {name}")
        return dict(row)
    finally:
        conn.close()


def update_student(name: str, **kwargs) -> bool:
    """Update student fields. Accepts any column name as keyword arg."""
    valid_fields = {
        "display_name", "grade", "age", "language", "difficulty",
        "notes", "favorite_subjects", "struggling_subjects"
    }
    updates = {k: v for k, v in kwargs.items() if k in valid_fields}
    if not updates:
        return False
    
    # JSON-encode lists
    for key in ("favorite_subjects", "struggling_subjects"):
        if key in updates and isinstance(updates[key], list):
            updates[key] = json.dumps(updates[key])
    
    conn = get_db()
    try:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [name]
        conn.execute(f"UPDATE students SET {set_clause} WHERE name = ?", values)
        conn.commit()
        return True
    finally:
        conn.close()


def get_student_profile(name: str) -> Optional[Dict]:
    """Get full student profile with computed fields."""
    conn = get_db()
    try:
        row = conn.execute("SELECT * FROM students WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        profile = dict(row)
        
        # Compute level
        profile["level"] = get_level(profile["total_points"])
        
        # Parse JSON fields
        for key in ("achievements", "languages_used", "favorite_subjects", "struggling_subjects"):
            try:
                profile[key] = json.loads(profile[key])
            except:
                profile[key] = []
        
        # Get recent subjects
        recent = conn.execute("""
            SELECT subject, COUNT(*) as cnt FROM interactions
            WHERE student_id = ? AND timestamp > datetime('now', '-7 days')
            GROUP BY subject ORDER BY cnt DESC LIMIT 5
        """, (profile["id"],)).fetchall()
        profile["recent_subjects"] = [{"subject": r["subject"], "count": r["cnt"]} for r in recent]
        
        # Get subject progress
        progress = conn.execute(
            "SELECT * FROM subject_progress WHERE student_id = ? ORDER BY mastery_score DESC",
            (profile["id"],)
        ).fetchall()
        profile["subject_progress"] = [dict(p) for p in progress]
        
        return profile
    finally:
        conn.close()


def get_all_students() -> List[Dict]:
    """Get all student profiles (summary)."""
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT name, display_name, grade, total_points, current_streak,
                   total_questions, last_seen, achievements
            FROM students ORDER BY last_seen DESC
        """).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["level"] = get_level(d["total_points"])
            try:
                d["achievements"] = json.loads(d["achievements"])
            except:
                d["achievements"] = []
            result.append(d)
        return result
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════
# Session Management
# ═══════════════════════════════════════════════════════

def start_session(student_name: str, mood: str = "") -> int:
    """Start a new study session. Returns session_id."""
    student = get_or_create_student(student_name)
    conn = get_db()
    try:
        cursor = conn.execute(
            "INSERT INTO sessions (student_id, mood_start) VALUES (?, ?)",
            (student["id"], mood)
        )
        session_id = cursor.lastrowid
        
        # Check for daily streak + first_today bonus
        today = datetime.now().strftime("%Y-%m-%d")
        points = 0
        if student["last_study_date"] != today:
            points += POINTS["first_today"]
            
            # Update streak
            yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            if student["last_study_date"] == yesterday:
                new_streak = student["current_streak"] + 1
            elif student["last_study_date"] == today:
                new_streak = student["current_streak"]
            else:
                new_streak = 1  # Streak broken
            
            longest = max(student["longest_streak"], new_streak)
            points += POINTS["streak_bonus"] * min(new_streak, 30)  # Cap streak bonus
            
            conn.execute("""
                UPDATE students SET last_study_date = ?, current_streak = ?,
                longest_streak = ?, total_points = total_points + ?
                WHERE id = ?
            """, (today, new_streak, longest, points, student["id"]))
            
            # Check streak achievements
            _check_streak_achievement(conn, student["id"], new_streak)
        
        # Check time-based achievements
        hour = datetime.now().hour
        if hour < 8:
            _grant_achievement(conn, student["id"], "early_bird")
        elif hour >= 21:
            _grant_achievement(conn, student["id"], "night_owl")
        
        conn.commit()
        return session_id
    finally:
        conn.close()


def end_session(session_id: int, mood: str = ""):
    """End a study session, compute duration."""
    conn = get_db()
    try:
        session = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not session:
            return
        
        started = datetime.fromisoformat(session["started_at"])
        duration = (datetime.now() - started).total_seconds() / 60.0
        
        conn.execute("""
            UPDATE sessions SET ended_at = CURRENT_TIMESTAMP,
            duration_minutes = ?, mood_end = ? WHERE id = ?
        """, (round(duration, 1), mood, session_id))
        
        # Update total study time
        conn.execute("""
            UPDATE students SET total_study_minutes = total_study_minutes + ?
            WHERE id = ?
        """, (round(duration), session["student_id"]))
        
        # Study time achievement
        student = conn.execute("SELECT total_study_minutes FROM students WHERE id = ?",
                               (session["student_id"],)).fetchone()
        if student and student["total_study_minutes"] >= 600:  # 10 hours
            _grant_achievement(conn, session["student_id"], "bookworm")
        
        # Study time points (every 5 min)
        time_points = int(duration / 5) * POINTS["study_5min"]
        if time_points > 0:
            conn.execute("UPDATE students SET total_points = total_points + ? WHERE id = ?",
                         (time_points, session["student_id"]))
        
        conn.commit()
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════
# Interaction Logging & Points
# ═══════════════════════════════════════════════════════

def log_interaction(student_name: str, session_id: int = None,
                    type: str = "question", subject: str = "general",
                    language: str = "el", question: str = "", answer: str = "",
                    was_correct: int = -1, emotion: str = "") -> Dict:
    """Log any interaction and award points. Returns points earned + any new achievements."""
    student = get_or_create_student(student_name)
    points = 0
    new_achievements = []
    
    conn = get_db()
    try:
        # Award points based on type
        if type == "question":
            points = POINTS["question_asked"]
        elif type == "quiz" and was_correct == 1:
            points = POINTS["quiz_correct"]
        elif type == "quiz" and was_correct == 0:
            points = POINTS["quiz_wrong"]
        elif type == "homework":
            points = POINTS["homework_scanned"]
        
        # Get student difficulty
        difficulty = student.get("difficulty", "normal")
        
        # Insert interaction
        conn.execute("""
            INSERT INTO interactions (student_id, session_id, type, subject, language,
                                     question, answer, was_correct, difficulty, points_earned, emotion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (student["id"], session_id, type, subject, language,
              question[:500], answer[:1000], was_correct, difficulty, points, emotion))
        
        # Update student totals
        conn.execute("""
            UPDATE students SET total_points = total_points + ?,
            total_questions = total_questions + 1 WHERE id = ?
        """, (points, student["id"]))
        
        # Update session
        if session_id:
            conn.execute("""
                UPDATE sessions SET questions_asked = questions_asked + 1,
                points_earned = points_earned + ? WHERE id = ?
            """, (points, session_id))
        
        # Track language
        langs = json.loads(student.get("languages_used", "[]"))
        if language not in langs:
            langs.append(language)
            conn.execute("UPDATE students SET languages_used = ? WHERE id = ?",
                         (json.dumps(langs), student["id"]))
            if len(langs) >= 2:
                if _grant_achievement(conn, student["id"], "polyglot"):
                    new_achievements.append("polyglot")
        
        # Update subject progress
        _update_subject_progress(conn, student["id"], subject, was_correct)
        
        # Check question milestones
        total_q = student["total_questions"] + 1
        if total_q == 1:
            if _grant_achievement(conn, student["id"], "first_question"):
                new_achievements.append("first_question")
        if total_q >= 100:
            if _grant_achievement(conn, student["id"], "curious_mind"):
                new_achievements.append("curious_mind")
        
        # Subject-specific achievements
        _check_subject_achievements(conn, student["id"], subject)
        
        # Level achievements
        new_total = student["total_points"] + points
        level = get_level(new_total)
        if level["number"] >= 5:
            if _grant_achievement(conn, student["id"], "level_5"):
                new_achievements.append("level_5")
        if level["number"] >= 10:
            if _grant_achievement(conn, student["id"], "level_10"):
                new_achievements.append("level_10")
        
        conn.commit()
        
        return {
            "points_earned": points,
            "total_points": new_total,
            "level": level,
            "new_achievements": [ACHIEVEMENTS[a] for a in new_achievements if a in ACHIEVEMENTS],
        }
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════
# Quiz System
# ═══════════════════════════════════════════════════════

def save_quiz_result(student_name: str, session_id: int = None,
                     subject: str = "", difficulty: str = "normal",
                     total_questions: int = 0, correct_answers: int = 0,
                     questions_json: list = None, time_taken: int = 0) -> Dict:
    """Save a completed quiz and award bonus points."""
    student = get_or_create_student(student_name)
    score = (correct_answers / total_questions * 100) if total_questions > 0 else 0
    
    points = POINTS["quiz_completed"]
    if score == 100:
        points += POINTS["perfect_quiz"]
    
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO quizzes (student_id, session_id, subject, difficulty,
                                total_questions, correct_answers, score_percent,
                                questions_json, time_taken_seconds, points_earned)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (student["id"], session_id, subject, difficulty,
              total_questions, correct_answers, score,
              json.dumps(questions_json or []), time_taken, points))
        
        conn.execute("""
            UPDATE students SET total_points = total_points + ?,
            total_quizzes = total_quizzes + 1,
            perfect_quizzes = perfect_quizzes + CASE WHEN ? = 100 THEN 1 ELSE 0 END
            WHERE id = ?
        """, (points, score, student["id"]))
        
        # Quiz master achievement (5 perfect quizzes)
        student_updated = conn.execute("SELECT perfect_quizzes FROM students WHERE id = ?",
                                        (student["id"],)).fetchone()
        new_achievements = []
        if student_updated and student_updated["perfect_quizzes"] >= 5:
            if _grant_achievement(conn, student["id"], "quiz_master"):
                new_achievements.append("quiz_master")
        
        conn.commit()
        
        return {
            "score_percent": score,
            "points_earned": points,
            "is_perfect": score == 100,
            "new_achievements": [ACHIEVEMENTS[a] for a in new_achievements if a in ACHIEVEMENTS],
        }
    finally:
        conn.close()


def get_quiz_history(student_name: str, limit: int = 10) -> List[Dict]:
    """Get recent quiz results for a student."""
    conn = get_db()
    try:
        student = conn.execute("SELECT id FROM students WHERE name = ?", (student_name,)).fetchone()
        if not student:
            return []
        rows = conn.execute("""
            SELECT * FROM quizzes WHERE student_id = ?
            ORDER BY timestamp DESC LIMIT ?
        """, (student["id"], limit)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════
# Homework Tracking
# ═══════════════════════════════════════════════════════

def log_homework(student_name: str, session_id: int = None,
                 subject: str = "", image_path: str = "",
                 extracted_text: str = "", explanation: str = "") -> Dict:
    """Log a homework scan."""
    student = get_or_create_student(student_name)
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO homework (student_id, session_id, subject, image_path,
                                  extracted_text, explanation)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (student["id"], session_id, subject, image_path,
              extracted_text[:2000], explanation[:2000]))
        
        conn.execute("""
            UPDATE students SET homework_scanned = homework_scanned + 1 WHERE id = ?
        """, (student["id"],))
        
        # Achievement
        new_achievements = []
        hw_count = conn.execute("SELECT homework_scanned FROM students WHERE id = ?",
                                (student["id"],)).fetchone()
        if hw_count and hw_count["homework_scanned"] >= 10:
            if _grant_achievement(conn, student["id"], "homework_hero"):
                new_achievements.append("homework_hero")
        
        conn.commit()
        return {"new_achievements": [ACHIEVEMENTS[a] for a in new_achievements if a in ACHIEVEMENTS]}
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════
# Smart Context for AI (Memory)
# ═══════════════════════════════════════════════════════

def get_student_context(student_name: str) -> str:
    """Generate a context string for the AI about this student.
    This is injected into the system prompt so VIRON 'remembers' the student."""
    profile = get_student_profile(student_name)
    if not profile:
        return ""
    
    parts = []
    parts.append(f"[STUDENT PROFILE: {profile['display_name']}]")
    
    if profile["grade"]:
        parts.append(f"Grade: {profile['grade']}")
    if profile["age"]:
        parts.append(f"Age: {profile['age']}")
    
    # Level & points
    level = profile["level"]
    parts.append(f"Level: {level['number']} ({level['name_el']}) — {profile['total_points']} points")
    parts.append(f"Streak: {profile['current_streak']} days")
    
    # Study stats
    if profile["total_questions"] > 0:
        parts.append(f"Questions asked: {profile['total_questions']}")
    if profile["total_study_minutes"] > 0:
        hours = profile["total_study_minutes"] // 60
        mins = profile["total_study_minutes"] % 60
        parts.append(f"Total study time: {hours}h {mins}m")
    
    # Difficulty
    parts.append(f"Difficulty: {profile['difficulty']}")
    
    # Subject strengths/weaknesses
    if profile["subject_progress"]:
        strong = [p["subject"] for p in profile["subject_progress"] if p["mastery_score"] > 0.7]
        weak = [p["subject"] for p in profile["subject_progress"] if p["mastery_score"] < 0.4]
        if strong:
            parts.append(f"Strong in: {', '.join(strong)}")
        if weak:
            parts.append(f"Needs help with: {', '.join(weak)}")
    
    # Recent activity
    if profile["recent_subjects"]:
        recent = ", ".join(f"{r['subject']}({r['count']})" for r in profile["recent_subjects"][:3])
        parts.append(f"Recent topics: {recent}")
    
    # Achievements (show last 3)
    if profile["achievements"]:
        recent_ach = profile["achievements"][-3:]
        ach_names = []
        for a_id in recent_ach:
            if a_id in ACHIEVEMENTS:
                ach_names.append(ACHIEVEMENTS[a_id]["icon"] + " " + ACHIEVEMENTS[a_id]["name_en"])
        if ach_names:
            parts.append(f"Recent achievements: {', '.join(ach_names)}")
    
    # Last seen
    if profile["last_seen"]:
        try:
            last = datetime.fromisoformat(profile["last_seen"])
            days_ago = (datetime.now() - last).days
            if days_ago == 0:
                parts.append("Last seen: today")
            elif days_ago == 1:
                parts.append("Last seen: yesterday")
            elif days_ago > 1:
                parts.append(f"Last seen: {days_ago} days ago")
        except:
            pass
    
    parts.append(f"Use {profile['display_name']}'s name naturally. Adapt difficulty to their level.")
    parts.append(f"If they've been away, welcome them back warmly.")
    
    return "\n".join(parts)


def get_greeting_context(student_name: str) -> Dict:
    """Get quick context for generating a personalized greeting."""
    profile = get_student_profile(student_name)
    if not profile:
        return {"is_new": True, "name": student_name}
    
    days_since = 0
    try:
        last = datetime.fromisoformat(profile["last_seen"])
        days_since = (datetime.now() - last).days
    except:
        pass
    
    return {
        "is_new": profile["total_questions"] == 0,
        "name": profile["display_name"],
        "points": profile["total_points"],
        "level": profile["level"],
        "streak": profile["current_streak"],
        "days_away": days_since,
        "total_questions": profile["total_questions"],
        "language": profile["language"],
    }


# ═══════════════════════════════════════════════════════
# Leaderboard
# ═══════════════════════════════════════════════════════

def get_leaderboard(limit: int = 10) -> List[Dict]:
    """Get top students by points."""
    conn = get_db()
    try:
        rows = conn.execute("""
            SELECT name, display_name, total_points, current_streak,
                   total_questions, total_quizzes, achievements
            FROM students ORDER BY total_points DESC LIMIT ?
        """, (limit,)).fetchall()
        result = []
        for i, row in enumerate(rows):
            d = dict(row)
            d["rank"] = i + 1
            d["level"] = get_level(d["total_points"])
            try:
                d["achievements"] = len(json.loads(d["achievements"]))
            except:
                d["achievements"] = 0
            result.append(d)
        return result
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════

def get_level(points: int) -> Dict:
    """Get level info for a given point total."""
    level_num = 1
    level_name_el = LEVELS[0][1]
    level_name_en = LEVELS[0][2]
    next_threshold = LEVELS[1][0] if len(LEVELS) > 1 else 99999
    
    for i, (threshold, name_el, name_en) in enumerate(LEVELS):
        if points >= threshold:
            level_num = i + 1
            level_name_el = name_el
            level_name_en = name_en
            next_threshold = LEVELS[i + 1][0] if i + 1 < len(LEVELS) else threshold
    
    return {
        "number": level_num,
        "name_el": level_name_el,
        "name_en": level_name_en,
        "points": points,
        "next_level_at": next_threshold,
        "progress": min(1.0, points / next_threshold) if next_threshold > 0 else 1.0,
    }


def _update_subject_progress(conn, student_id: int, subject: str, was_correct: int):
    """Update subject mastery score."""
    if subject == "general" or subject == "greeting":
        return
    
    row = conn.execute(
        "SELECT * FROM subject_progress WHERE student_id = ? AND subject = ?",
        (student_id, subject)
    ).fetchone()
    
    if row:
        new_asked = row["questions_asked"] + 1
        new_correct = row["correct_answers"] + (1 if was_correct == 1 else 0)
        # Mastery = weighted recent accuracy (simple moving average)
        mastery = new_correct / new_asked if new_asked > 0 else 0.0
        conn.execute("""
            UPDATE subject_progress SET questions_asked = ?, correct_answers = ?,
            mastery_score = ?, updated_at = CURRENT_TIMESTAMP
            WHERE student_id = ? AND subject = ?
        """, (new_asked, new_correct, round(mastery, 3), student_id, subject))
    else:
        mastery = 1.0 if was_correct == 1 else 0.0 if was_correct == 0 else 0.5
        conn.execute("""
            INSERT INTO subject_progress (student_id, subject, questions_asked,
                                          correct_answers, mastery_score)
            VALUES (?, ?, 1, ?, ?)
        """, (student_id, subject, 1 if was_correct == 1 else 0, mastery))


def _grant_achievement(conn, student_id: int, achievement_id: str) -> bool:
    """Grant an achievement if not already earned. Returns True if new."""
    if achievement_id not in ACHIEVEMENTS:
        return False
    
    row = conn.execute("SELECT achievements FROM students WHERE id = ?", (student_id,)).fetchone()
    if not row:
        return False
    
    try:
        current = json.loads(row["achievements"])
    except:
        current = []
    
    if achievement_id in current:
        return False  # Already earned
    
    current.append(achievement_id)
    conn.execute("UPDATE students SET achievements = ? WHERE id = ?",
                 (json.dumps(current), student_id))
    
    ach = ACHIEVEMENTS[achievement_id]
    print(f"  🏆 Achievement unlocked: {ach['icon']} {ach['name_en']}")
    return True


def _check_streak_achievement(conn, student_id: int, streak: int):
    """Check and grant streak achievements."""
    if streak >= 3:
        _grant_achievement(conn, student_id, "streak_3")
    if streak >= 7:
        _grant_achievement(conn, student_id, "streak_7")
    if streak >= 30:
        _grant_achievement(conn, student_id, "streak_30")


def _check_subject_achievements(conn, student_id: int, subject: str):
    """Check subject-specific achievements."""
    if subject in ("math", "mathematics"):
        row = conn.execute("""
            SELECT COUNT(*) as cnt FROM interactions
            WHERE student_id = ? AND subject IN ('math', 'mathematics') AND was_correct = 1
        """, (student_id,)).fetchone()
        if row and row["cnt"] >= 50:
            _grant_achievement(conn, student_id, "math_whiz")
    
    elif subject == "science":
        row = conn.execute("""
            SELECT COUNT(*) as cnt FROM interactions
            WHERE student_id = ? AND subject = 'science'
        """, (student_id,)).fetchone()
        if row and row["cnt"] >= 30:
            _grant_achievement(conn, student_id, "science_explorer")


# ═══════════════════════════════════════════════════════
# Initialize on import
# ═══════════════════════════════════════════════════════

init_db()
