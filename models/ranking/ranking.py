import os
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

class XGBoostRanker:
    def __init__(self, model_dir="models/ranking"):
        self.model_dir = model_dir
        self.model = None
        self.feature_names = [
            "two_tower_similarity",
            "svd_prediction",
            "content_similarity",
            "genre_match_score",
            "author_match_score",
            "book_average_rating",
            "book_popularity",
            "book_reviews_count"
        ]
        os.makedirs(self.model_dir, exist_ok=True)
        
    def _calculate_features(self, user_row, book_row, user_embedding, book_embeddings, 
                             svd_recommender, content_recommender, user_pos_books_tfidf):
        # 1. Two-tower similarity
        # Retrieve two-tower indices from mappings
        # (embeddings are offset by 1 because of padding in PyTorch)
        two_tower_mappings = content_recommender.two_tower_mappings if hasattr(content_recommender, "two_tower_mappings") else None
        
        sim = 0.0
        if two_tower_mappings is not None and user_embedding is not None and book_embeddings is not None:
            try:
                u_idx = two_tower_mappings["user_encoder"].transform([user_row["user_id"]])[0] + 1
                b_idx = two_tower_mappings["book_encoder"].transform([book_row["book_id"]])[0] + 1
                
                u_emb = user_embedding[u_idx]
                b_emb = book_embeddings[b_idx]
                sim = float(np.dot(u_emb, b_emb))
            except Exception:
                pass
                
        # 2. SVD prediction
        svd_pred = svd_recommender.predict_rating(user_row["user_id"], book_row["book_id"], model_type="svd")
        
        # 3. Content TF-IDF similarity
        content_sim = 0.0
        if user_pos_books_tfidf is not None and user_pos_books_tfidf.shape[0] > 0:
            try:
                book_idx = content_recommender.indices_by_id.get(book_row["book_id"])
                if book_idx is not None:
                    book_tfidf = content_recommender.tfidf_matrix[book_idx]
                    # Calculate similarity between this book and user's high-rated books
                    sims = (user_pos_books_tfidf * book_tfidf.T).toarray().flatten()
                    content_sim = float(np.max(sims))
            except Exception:
                pass
                
        # 4. Genre Jaccard Match
        user_genres = set(user_row["favorite_genres"].split("|")) if isinstance(user_row["favorite_genres"], str) else set()
        book_genres = set(book_row["standardized_genres"].split("|")) if isinstance(book_row["standardized_genres"], str) else set()
        
        union = user_genres.union(book_genres)
        intersection = user_genres.intersection(book_genres)
        genre_match = len(intersection) / len(union) if union else 0.0
        
        # 5. Author Match
        user_authors = set(user_row["favorite_authors"].split("|")) if isinstance(user_row["favorite_authors"], str) else set()
        author_match = 1.0 if book_row["author"].lower() in user_authors else 0.0
        
        return {
            "two_tower_similarity": sim,
            "svd_prediction": svd_pred,
            "content_similarity": content_sim,
            "genre_match_score": genre_match,
            "author_match_score": author_match,
            "book_average_rating": float(book_row["average_rating"]),
            "book_popularity": float(book_row["ratings_count"]),
            "book_reviews_count": float(book_row["reviews_count"])
        }
        
    def generate_features_dataset(self, df_interactions, df_books, df_users, 
                                  svd_recommender, content_recommender,
                                  user_embeddings, book_embeddings, two_tower_mappings):
        print("Generating feature vectors for XGBoost training...")
        
        # Bind mappings to content recommender temporarily for easier lookup
        content_recommender.two_tower_mappings = two_tower_mappings
        
        # Build lookups
        user_lookup = {row["user_id"]: row for _, row in df_users.iterrows()}
        book_lookup = {row["book_id"]: row for _, row in df_books.iterrows()}
        
        # Find positive books per user for content similarity feature
        user_pos_books = df_interactions[
            (df_interactions["rating"] >= 4) | (df_interactions["interaction_type"].isin(["like", "shelve"]))
        ].groupby("user_id")["book_id"].apply(list).to_dict()
        
        # Pre-extract user TF-IDF representations for efficiency
        user_tfidfs = {}
        for u_id, b_ids in user_pos_books.items():
            indices = [content_recommender.indices_by_id[b] for b in b_ids if b in content_recommender.indices_by_id]
            if indices:
                user_tfidfs[u_id] = content_recommender.tfidf_matrix[indices]
                
        features_list = []
        labels_list = []
        
        for idx, row in df_interactions.iterrows():
            u_id = row["user_id"]
            b_id = row["book_id"]
            
            if u_id not in user_lookup or b_id not in book_lookup:
                continue
                
            u_row = user_lookup[u_id]
            b_row = book_lookup[b_id]
            
            # Label definition: rating >= 4 or positive interaction
            label = 0.0
            if pd.notna(row["rating"]):
                label = 1.0 if row["rating"] >= 4.0 else 0.0
            else:
                label = 1.0 if row["interaction_type"] in ["like", "shelve"] else 0.0
                
            feat_dict = self._calculate_features(
                u_row, b_row, user_embeddings, book_embeddings,
                svd_recommender, content_recommender, user_tfidfs.get(u_id)
            )
            
            features_list.append(feat_dict)
            labels_list.append(label)
            
        df_feat = pd.DataFrame(features_list)
        return df_feat, np.array(labels_list)
        
    def fit(self, X_train, y_train, X_val, y_val):
        print("Training XGBoost Classifier...")
        
        self.model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=150,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            early_stopping_rounds=15
        )
        
        self.model.fit(
            X_train[self.feature_names], y_train,
            eval_set=[(X_val[self.feature_names], y_val)],
            verbose=True
        )
        
        # Evaluate
        train_preds = self.model.predict_proba(X_train[self.feature_names])[:, 1]
        val_preds = self.model.predict_proba(X_val[self.feature_names])[:, 1]
        
        train_auc = roc_auc_score(y_train, train_preds)
        val_auc = roc_auc_score(y_val, val_preds)
        
        train_loss = log_loss(y_train, train_preds)
        val_loss = log_loss(y_val, val_preds)
        
        print(f"\n--- XGBoost Training Metrics ---")
        print(f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
        print(f"Train LogLoss: {train_loss:.4f} | Val LogLoss: {val_loss:.4f}")
        
        # Save metrics to reports
        os.makedirs("reports", exist_ok=True)
        with open("reports/ranking_xgboost_report.txt", "w") as f:
            f.write("=== XGBOOST CANDIDATE RANKER EVALUATION REPORT ===\n\n")
            f.write(f"Train AUC:     {train_auc:.4f}\n")
            f.write(f"Validation AUC: {val_auc:.4f}\n")
            f.write(f"Train LogLoss: {train_loss:.4f}\n")
            f.write(f"Val LogLoss:   {val_loss:.4f}\n\n")
            
            f.write("Feature Importances:\n")
            importances = self.model.feature_importances_
            for name, imp in zip(self.feature_names, importances):
                f.write(f"  {name}: {imp:.4f}\n")
                
    def save(self):
        print("Saving XGBoost model...")
        with open(os.path.join(self.model_dir, "xgboost_ranker.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        print("Saved successfully.")
        
    def load(self):
        print("Loading XGBoost model...")
        model_path = os.path.join(self.model_dir, "xgboost_ranker.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError("XGBoost model not found. Train the model first.")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        print("Loaded successfully.")
        
    def rank_candidates(self, user_row, candidate_books_df, user_embedding, book_embeddings,
                        svd_recommender, content_recommender, user_pos_books_tfidf, top_n=10):
        if self.model is None:
            raise ValueError("XGBoost model not loaded.")
            
        features_list = []
        for _, b_row in candidate_books_df.iterrows():
            feat_dict = self._calculate_features(
                user_row, b_row, user_embedding, book_embeddings,
                svd_recommender, content_recommender, user_pos_books_tfidf
            )
            features_list.append(feat_dict)
            
        df_feat = pd.DataFrame(features_list)
        
        # Run prediction
        probs = self.model.predict_proba(df_feat[self.feature_names])[:, 1]
        
        # Sort candidates
        ranked_indices = np.argsort(probs)[::-1]
        
        ranked_books = []
        for idx in ranked_indices[:top_n]:
            b_row = candidate_books_df.iloc[idx]
            prob = probs[idx]
            
            # Map features for explainability
            feats = features_list[idx]
            
            ranked_books.append({
                "book_id": int(b_row["book_id"]),
                "title": b_row["title"],
                "author": b_row["author"],
                "genres": b_row["genres"],
                "score": float(prob),
                "features": feats # save for explanation engine
            })
            
        return ranked_books
