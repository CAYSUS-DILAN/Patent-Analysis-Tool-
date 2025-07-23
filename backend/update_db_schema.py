import sqlite3

def update_db_schema():
    db = sqlite3.connect('users.db')
    cursor = db.cursor()

    # Check if the role column exists
    cursor.execute("PRAGMA table_info(users)")
    columns = [column[1] for column in cursor.fetchall()]

    if 'role' not in columns:
        # Add the role column if it doesn't exist
        cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
        db.commit()
        print("Added 'role' column to 'users' table.")
    else:
        print("'role' column already exists in 'users' table.")

    db.close()

update_db_schema()
