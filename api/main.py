import os
import sys
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.db import get_db, init_db
from database.models import Book, User, Rating, UserPreference, RecommendationLog
from models.hybrid import HybridRecommendationEngine

# Initialize FastAPI App
app = FastAPI(
    title="AI Book Recommendation System API",
    description="Production-Grade Hybrid Recommendation Engine API simulating Netflix/Spotify models.",
    version="1.0.0"
)

# Global recommendation engine instance
engine = None

@app.on_event("startup")
def startup_event():
    global engine
    # Initialize DB schema and seed if necessary
    init_db()
    
    # Load recommendation models
    engine = HybridRecommendationEngine()
    try:
        engine.load_models()
        print("FastAPI: Recommendation models loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load recommendation models. Run train_pipeline.py. Error: {e}")

# --- Pydantic Schemas ---

class RecommendationItem(BaseModel):
    book_id: int
    title: str
    author: str
    genres: str
    score: float
    reason: str

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendationItem]

class BookResponse(BaseModel):
    book_id: int
    title: str
    author: str
    genres: str
    description: str
    average_rating: float
    ratings_count: int
    reviews_count: int

    class Config:
        from_attributes = True

class RateBookRequest(BaseModel):
    user_id: int
    book_id: int
    rating: Optional[float] = Field(None, ge=1.0, le=5.0)
    interaction_type: str = Field("rate", description="Type of interaction: rate, view, like, shelve")
    review_text: Optional[str] = None

class NewUserRequest(BaseModel):
    username: str
    favorite_genres: List[str]
    favorite_authors: List[str]

class NewUserResponse(BaseModel):
    user_id: int
    username: str
    message: str

class HealthResponse(BaseModel):
    status: str
    database: str
    models_loaded: bool

# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check(db: Session = Depends(get_db)):
    db_status = "healthy"
    try:
        # Check connection
        db.execute(BaseModel.metadata.tables.get("users").select().limit(1))
    except Exception:
        # SQLite or PostgreSQL check
        try:
            db.query(User).first()
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
            
    models_loaded = engine is not None and engine.xgboost_ranker.model is not None
    
    return {
        "status": "healthy" if db_status == "healthy" and models_loaded else "degraded",
        "database": db_status,
        "models_loaded": models_loaded
    }

@app.get("/recommend/{user_id}", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(user_id: int, top_n: int = 10, db: Session = Depends(get_db)):
    if engine is None or engine.xgboost_ranker.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation models are not loaded. Train the models first."
        )
        
    user = db.query(User).filter(User.id == user_id).first()
    
    # Check if user exists in database. If not, raise 404
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User ID {user_id} not found. Register user using POST /new-user first."
        )
        
    try:
        # Fetch hybrid recommendations
        recs = engine.recommend(user_id=user_id, top_n=top_n)
        
        # Log recommendations in database
        book_ids_str = ",".join([str(r["book_id"]) for r in recs])
        log = RecommendationLog(
            user_id=user_id,
            recommended_book_ids=book_ids_str,
            weights_used="default"
        )
        db.add(log)
        db.commit()
        
        return {"recommendations": recs}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate recommendations: {str(e)}"
        )

@app.get("/book/{book_id}", response_model=BookResponse, tags=["Catalog"])
def get_book_details(book_id: int, db: Session = Depends(get_db)):
    book = db.query(Book).filter(Book.book_id == book_id).first()
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book ID {book_id} not found."
        )
    return book

@app.get("/similar-books/{book_id}", response_model=List[RecommendationItem], tags=["Catalog"])
def get_similar_books(book_id: int, top_n: int = 10, db: Session = Depends(get_db)):
    if engine is None or engine.content_rec.tfidf_matrix is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Content-based similarity model not loaded."
        )
        
    book = db.query(Book).filter(Book.book_id == book_id).first()
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book ID {book_id} not found."
        )
        
    recs = engine.content_rec.recommend_similar_books(book_id=book_id, top_n=top_n)
    return recs

@app.post("/rate-book", status_code=status.HTTP_201_CREATED, tags=["Interactions"])
def rate_book(request: RateBookRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User ID {request.user_id} not found."
        )
        
    book = db.query(Book).filter(Book.book_id == request.book_id).first()
    if not book:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Book ID {request.book_id} not found."
        )
        
    # Check if interaction already exists
    existing_rating = db.query(Rating).filter(
        Rating.user_id == request.user_id,
        Rating.book_id == request.book_id
    ).first()
    
    try:
        if existing_rating:
            # Update rating
            existing_rating.rating = request.rating
            existing_rating.interaction_type = request.interaction_type
            existing_rating.review_text = request.review_text
            message = "Interaction updated successfully."
        else:
            # Create new rating
            new_rating = Rating(
                user_id=request.user_id,
                book_id=request.book_id,
                rating=request.rating,
                interaction_type=request.interaction_type,
                review_text=request.review_text
            )
            db.add(new_rating)
            message = "Interaction recorded successfully."
            
        db.commit()
        return {"message": message}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record interaction: {str(e)}"
        )

@app.post("/new-user", response_model=NewUserResponse, status_code=status.HTTP_201_CREATED, tags=["Users"])
def create_new_user(request: NewUserRequest, db: Session = Depends(get_db)):
    # Check if username exists
    existing_user = db.query(User).filter(User.username == request.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Username '{request.username}' is already taken."
        )
        
    try:
        # Create user
        # Auto-increment ID
        max_id = db.query(User).order_by(User.id.desc()).first()
        new_id = (max_id.id + 1) if max_id else 1
        
        new_user = User(id=new_id, username=request.username)
        db.add(new_user)
        db.flush() # Flush to get user id in session
        
        # Create user preferences
        genres_str = "|".join([g.lower().strip() for g in request.favorite_genres])
        authors_str = "|".join([a.lower().strip() for a in request.favorite_authors])
        
        prefs = UserPreference(
            user_id=new_id,
            favorite_genres=genres_str,
            favorite_authors=authors_str
        )
        db.add(prefs)
        db.commit()
        
        return {
            "user_id": new_id,
            "username": request.username,
            "message": "User registered successfully."
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register user: {str(e)}"
        )
