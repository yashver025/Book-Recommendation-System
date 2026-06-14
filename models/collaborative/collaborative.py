import os
import pickle
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic, SVD, accuracy

class CollaborativeRecommender:
    def __init__(self, model_dir="models/collaborative"):
        self.model_dir = model_dir
        self.svd_model = None
        self.knn_model = None
        self.trainset = None
        os.makedirs(self.model_dir, exist_ok=True)
        
    def fit(self, train_df):
        print("Training Collaborative Filtering models (SVD & KNNBasic)...")
        # Define Reader format for rating values (min 1, max 5)
        reader = Reader(rating_scale=(1.0, 5.0))
        
        # Surprise needs a df with [user_id, item_id, rating]
        surprise_df = train_df[["user_id", "book_id", "rating"]].dropna()
        
        # Load dataset
        data = Dataset.load_from_df(surprise_df, reader)
        self.trainset = data.build_full_trainset()
        
        # 1. Train SVD
        print("Fitting SVD model...")
        self.svd_model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
        self.svd_model.fit(self.trainset)
        
        # 2. Train KNNBasic (user-based)
        print("Fitting KNNBasic model...")
        sim_options = {"name": "cosine", "user_based": True}
        self.knn_model = KNNBasic(sim_options=sim_options, random_state=42, verbose=False)
        self.knn_model.fit(self.trainset)
        
        print("Training complete.")
        
    def save(self):
        print("Saving Collaborative Filtering models...")
        with open(os.path.join(self.model_dir, "svd_model.pkl"), "wb") as f:
            pickle.dump(self.svd_model, f)
        with open(os.path.join(self.model_dir, "knn_model.pkl"), "wb") as f:
            pickle.dump(self.knn_model, f)
        print("Saved successfully.")
        
    def load(self):
        print("Loading Collaborative Filtering models...")
        svd_path = os.path.join(self.model_dir, "svd_model.pkl")
        knn_path = os.path.join(self.model_dir, "knn_model.pkl")
        
        if not (os.path.exists(svd_path) and os.path.exists(knn_path)):
            raise FileNotFoundError("Collaborative models not found. Fit the models first.")
            
        with open(svd_path, "rb") as f:
            self.svd_model = pickle.load(f)
        with open(knn_path, "rb") as f:
            self.knn_model = pickle.load(f)
        print("Loaded successfully.")
        
    def predict_rating(self, user_id, book_id, model_type="svd"):
        if model_type == "svd":
            if self.svd_model is None:
                raise ValueError("SVD model not loaded.")
            pred = self.svd_model.predict(str(user_id), str(book_id))
            # If the user or book was unseen, Surprise handles it by returning the global mean
            # However, we must ensure user_id and book_id are passed in their training formats (strings or ints)
            # Surprise's internal mappings might use strings if columns were loaded as strings.
            # Let's try matching types
            if pred.details.get("was_impossible", False):
                # Retry with int
                pred = self.svd_model.predict(int(user_id), int(book_id))
            return pred.est
        elif model_type == "knn":
            if self.knn_model is None:
                raise ValueError("KNNBasic model not loaded.")
            pred = self.knn_model.predict(str(user_id), str(book_id))
            if pred.details.get("was_impossible", False):
                pred = self.knn_model.predict(int(user_id), int(book_id))
            return pred.est
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def recommend_for_user(self, user_id, train_df, processed_books_df, top_n=10):
        # Find books already rated by the user
        user_ratings = train_df[train_df["user_id"] == user_id]
        rated_book_ids = set(user_ratings["book_id"].unique())
        
        # Get all catalog book IDs
        all_book_ids = processed_books_df["book_id"].unique()
        
        # Predict ratings for unrated books
        predictions = []
        for b_id in all_book_ids:
            if b_id not in rated_book_ids:
                pred_rating = self.predict_rating(user_id, b_id, model_type="svd")
                predictions.append((b_id, pred_rating))
                
        # Sort predictions by score descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for b_id, score in predictions[:top_n]:
            book_row = processed_books_df[processed_books_df["book_id"] == b_id].iloc[0]
            recommendations.append({
                "book_id": int(b_id),
                "title": book_row["title"],
                "author": book_row["author"],
                "genres": book_row["genres"],
                "score": float(score),
                "reason": "Recommended because readers with similar profiles rated this highly."
            })
            
        return recommendations
        
    def evaluate_models(self, val_df):
        print("Evaluating SVD and KNNBasic models on validation set...")
        reader = Reader(rating_scale=(1.0, 5.0))
        
        # Prepare validation testset format for surprise
        val_df_clean = val_df[["user_id", "book_id", "rating"]].dropna()
        val_data = Dataset.load_from_df(val_df_clean, reader)
        
        # Build surprise test set: list of (uid, iid, r_ui)
        testset = [
            (row["user_id"], row["book_id"], row["rating"])
            for _, row in val_df_clean.iterrows()
        ]
        
        # Predict and evaluate SVD
        svd_preds = self.svd_model.test(testset)
        svd_rmse = accuracy.rmse(svd_preds, verbose=False)
        svd_mae = accuracy.mae(svd_preds, verbose=False)
        
        # Predict and evaluate KNN
        knn_preds = self.knn_model.test(testset)
        knn_rmse = accuracy.rmse(knn_preds, verbose=False)
        knn_mae = accuracy.mae(knn_preds, verbose=False)
        
        report_path = "reports/collaborative_report.txt"
        os.makedirs("reports", exist_ok=True)
        with open(report_path, "w") as f:
            f.write("=== COLLABORATIVE FILTERING EVALUATION REPORT ===\n\n")
            f.write(f"Model SVD:\n  RMSE: {svd_rmse:.4f}\n  MAE:  {svd_mae:.4f}\n\n")
            f.write(f"Model KNNBasic:\n  RMSE: {knn_rmse:.4f}\n  MAE:  {knn_mae:.4f}\n\n")
            
        print(f"Evaluation report written to {report_path}")
        return {
            "svd": {"rmse": svd_rmse, "mae": svd_mae},
            "knn": {"rmse": knn_rmse, "mae": knn_mae}
        }

if __name__ == "__main__":
    if os.path.exists("data/train_interactions.csv") and os.path.exists("data/val_interactions.csv"):
        df_train = pd.read_csv("data/train_interactions.csv")
        df_val = pd.read_csv("data/val_interactions.csv")
        df_books = pd.read_csv("data/processed_books.csv")
        
        recommender = CollaborativeRecommender()
        recommender.fit(df_train)
        recommender.save()
        
        metrics = recommender.evaluate_models(df_val)
        print(f"SVD RMSE: {metrics['svd']['rmse']:.4f}")
        print(f"KNN RMSE: {metrics['knn']['rmse']:.4f}")
