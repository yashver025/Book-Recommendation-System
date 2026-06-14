import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base, Book, User, Rating, UserPreference

# Database configuration
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

# Choose SQLite or PostgreSQL based on environment variables
if DB_USER and DB_PASSWORD and DB_HOST and DB_NAME:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(f"Database: Configured to use PostgreSQL at {DB_HOST}:{DB_PORT}")
else:
    os.makedirs("data", exist_ok=True)
    DATABASE_URL = "sqlite:///data/recsys.db"
    print("Database: Environment not fully configured for PostgreSQL. Falling back to local SQLite at data/recsys.db")

# Create engine and session maker
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    print("Initializing database schemas...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Check if books are already seeded
        if db.query(Book).count() == 0:
            print("Seeding books from data/processed_books.csv...")
            if os.path.exists("data/processed_books.csv"):
                df_books = pd.read_csv("data/processed_books.csv")
                books = []
                for _, row in df_books.iterrows():
                    books.append(Book(
                        book_id=int(row["book_id"]),
                        title=row["title"],
                        author=row["author"],
                        genres=row["genres"],
                        description=row["description"],
                        average_rating=float(row["average_rating"]),
                        ratings_count=int(row["ratings_count"]),
                        reviews_count=int(row["reviews_count"])
                    ))
                db.bulk_save_objects(books)
                db.commit()
                print(f"Successfully seeded {len(books)} books.")
            else:
                print("Warning: data/processed_books.csv not found. Skipping books seeding.")
                
        # Check if users are already seeded
        if db.query(User).count() == 0:
            print("Seeding users from data/processed_users.csv...")
            if os.path.exists("data/processed_users.csv"):
                df_users = pd.read_csv("data/processed_users.csv")
                users = []
                preferences = []
                for _, row in df_users.iterrows():
                    u_id = int(row["user_id"])
                    users.append(User(
                        id=u_id,
                        username=row["username"]
                    ))
                    # Add user preferences
                    preferences.append(UserPreference(
                        user_id=u_id,
                        favorite_genres=row["favorite_genres"],
                        favorite_authors=row["favorite_authors"]
                    ))
                db.bulk_save_objects(users)
                db.bulk_save_objects(preferences)
                db.commit()
                print(f"Successfully seeded {len(users)} users and preferences.")
            else:
                print("Warning: data/processed_users.csv not found. Skipping users seeding.")
                
        # Check if ratings are already seeded
        if db.query(Rating).count() == 0:
            print("Seeding ratings from data/interactions.csv...")
            if os.path.exists("data/interactions.csv"):
                df_ratings = pd.read_csv("data/interactions.csv")
                ratings = []
                # To prevent excessive SQLite overhead, we bulk insert in chunks
                for _, row in df_ratings.iterrows():
                    ratings.append(Rating(
                        user_id=int(row["user_id"]),
                        book_id=int(row["book_id"]),
                        rating=float(row["rating"]) if pd.notna(row["rating"]) else None,
                        interaction_type=row["interaction_type"],
                        review_text=row["review_text"] if pd.notna(row["review_text"]) else None
                    ))
                db.bulk_save_objects(ratings)
                db.commit()
                print(f"Successfully seeded {len(ratings)} interactions.")
            else:
                print("Warning: data/interactions.csv not found. Skipping ratings seeding.")
                
    except Exception as e:
        print(f"Error seeding database: {e}")
        db.rollback()
    finally:
        db.close()
        print("Database initialization complete.")
