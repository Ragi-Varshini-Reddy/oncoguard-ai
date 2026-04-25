import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.db.db import initialize_database

if __name__ == "__main__":
    print("🚀 Initializing Oncoguard-AI MySQL Database...")
    try:
        initialize_database()
        print("✅ Database initialized and seeded successfully.")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)
