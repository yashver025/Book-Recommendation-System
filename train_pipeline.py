import os
import sys
import pickle
import numpy as np
import pandas as pd

# Add the project root to path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.content_based.recommender import ContentBasedRecommender
from models.collaborative.collaborative import CollaborativeRecommender
from models.two_tower.train import train_two_tower
from faiss_retrieval.faiss_index import FaissRetriever, run_faiss_benchmark
from models.ranking.ranking import XGBoostRanker
from database.db import init_db

def main():
    print("====================================================")
    print("STARTING BOOK RECOMMENDATION SYSTEM TRAINING PIPELINE")
    print("====================================================\n")
    
    # 0. Check data availability
    if not (os.path.exists("data/processed_books.csv") and os.path.exists("data/train_interactions.csv")):
        print("Processed datasets missing. Run data engineering scripts first.")
        # Attempt to run them
        import subprocess
        print("Attempting to run data generation & preprocessing...")
        subprocess.run(["python", "preprocessing/data_generator.py"], check=True)
        subprocess.run(["python", "preprocessing/preprocess.py"], check=True)
        
    df_books = pd.read_csv("data/processed_books.csv")
    df_users = pd.read_csv("data/processed_users.csv")
    df_train = pd.read_csv("data/train_interactions.csv")
    df_val = pd.read_csv("data/val_interactions.csv")
    
    # 1. Fit Content-Based Recommender
    print("\n--- [1/6] Training Content-Based TF-IDF Recommender ---")
    content_rec = ContentBasedRecommender()
    content_rec.fit(df_books)
    content_rec.save()
    
    # 2. Train Collaborative Filtering (SVD & KNN)
    print("\n--- [2/6] Training Collaborative Filtering Models ---")
    collab_rec = CollaborativeRecommender()
    collab_rec.fit(df_train)
    collab_rec.save()
    collab_rec.evaluate_models(df_val)
    
    # 3. Train Two-Tower Deep Learning Model
    print("\n--- [3/6] Training Deep Learning Two-Tower Model ---")
    # This generates user_embeddings.npy and book_embeddings.npy
    train_two_tower()
    
    # 4. Build FAISS Retrieval Layer & Run Benchmark
    print("\n--- [4/6] Building FAISS Index & Benchmarking Vector Search ---")
    retriever = FaissRetriever()
    retriever.build_index()
    # Run the benchmark
    try:
        run_faiss_benchmark()
    except Exception as e:
        print(f"Error running FAISS benchmark: {e}")
        
    # 5. Train XGBoost Ranker
    print("\n--- [5/6] Training Candidate Ranker (XGBoost) ---")
    # Load newly generated embeddings
    user_embeddings = np.load("models/two_tower/user_embeddings.npy")
    book_embeddings = np.load("models/two_tower/book_embeddings.npy")
    with open("models/two_tower/two_tower_mappings.pkl", "rb") as f:
        two_tower_mappings = pickle.load(f)
        
    # Load fitted recommender engines
    content_rec.load()
    collab_rec.load()
    
    xgboost_ranker = XGBoostRanker()
    
    # Generate XGBoost features
    X_train, y_train = xgboost_ranker.generate_features_dataset(
        df_train, df_books, df_users, collab_rec, content_rec,
        user_embeddings, book_embeddings, two_tower_mappings
    )
    X_val, y_val = xgboost_ranker.generate_features_dataset(
        df_val, df_books, df_users, collab_rec, content_rec,
        user_embeddings, book_embeddings, two_tower_mappings
    )
    
    # Fit & Save XGBoost Ranker
    xgboost_ranker.fit(X_train, y_train, X_val, y_val)
    xgboost_ranker.save()
    
    # 6. Initialize & Seed Database
    print("\n--- [6/6] Initializing & Seeding SQL Database ---")
    init_db()
    
    print("\n====================================================")
    print("TRAINING PIPELINE COMPLETE! ALL MODEL ARTIFACTS SAVED")
    print("====================================================")

if __name__ == "__main__":
    main()
