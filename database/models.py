from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    ratings = relationship("Rating", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", uselist=False, back_populates="user", cascade="all, delete-orphan")
    logs = relationship("RecommendationLog", back_populates="user", cascade="all, delete-orphan")

class Book(Base):
    __tablename__ = "books"
    
    book_id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False, index=True)
    author = Column(String(255), nullable=False, index=True)
    genres = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    average_rating = Column(Float, default=0.0)
    ratings_count = Column(Integer, default=0)
    reviews_count = Column(Integer, default=0)
    
    # Relationships
    ratings = relationship("Rating", back_populates="book", cascade="all, delete-orphan")

class Rating(Base):
    __tablename__ = "ratings"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    book_id = Column(Integer, ForeignKey("books.book_id"), nullable=False, index=True)
    rating = Column(Float, nullable=True) # Can be null for implicit interactions
    interaction_type = Column(String(50), default="rate") # 'rate', 'view', 'like', 'shelve'
    review_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="ratings")
    book = relationship("Book", back_populates="ratings")

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    favorite_genres = Column(String(500), nullable=True) # pipeseparated list e.g. "fantasy|romance"
    favorite_authors = Column(String(500), nullable=True) # pipeseparated list
    
    # Relationships
    user = relationship("User", back_populates="preferences")

class RecommendationLog(Base):
    __tablename__ = "recommendation_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    recommended_book_ids = Column(String(500), nullable=False) # commaseparated list of book IDs
    weights_used = Column(String(200), nullable=True) # JSON or config string
    recommended_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="logs")
