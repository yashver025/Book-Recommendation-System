import os
import pickle
import numpy as np
import pandas as pd
from models.content_based.recommender import ContentBasedRecommender
from models.collaborative.collaborative import CollaborativeRecommender
from faiss_retrieval.faiss_index import FaissRetriever
from models.ranking.ranking import XGBoostRanker

class HybridRecommendationEngine:
    def __init__(self, data_dir="data", models_dir="models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        
        # Initialize sub-recommenders
        self.content_rec = ContentBasedRecommender(model_dir=os.path.join(models_dir, "content_based"))
        self.collab_rec = CollaborativeRecommender(model_dir=os.path.join(models_dir, "collaborative"))
        self.faiss_retriever = FaissRetriever(
            index_path=os.path.join(models_dir, "two_tower", "faiss_index.bin"),
            mappings_path=os.path.join(models_dir, "two_tower", "two_tower_mappings.pkl")
        )
        self.xgboost_ranker = XGBoostRanker(model_dir=os.path.join(models_dir, "ranking"))
        
        # Mappings & Embeddings
        self.two_tower_mappings = None
        self.user_embeddings = None
        self.book_embeddings = None
        
        # Dataframes
        self.df_books = None
        self.df_users = None
        self.df_interactions = None
        
    def load_models(self):
        print("Loading all sub-models in Hybrid Recommendation Engine...")
        
        # Load raw/processed datasets
        self.df_books = pd.read_csv(os.path.join(self.data_dir, "processed_books.csv"))
        self.df_users = pd.read_csv(os.path.join(self.data_dir, "processed_users.csv"))
        self.df_interactions = pd.read_csv(os.path.join(self.data_dir, "interactions.csv"))
        
        # Load content recommender
        self.content_rec.load()
        
        # Load collaborative recommender
        self.collab_rec.load()
        
        # Load Two-Tower mappings & embeddings
        mappings_path = os.path.join(self.models_dir, "two_tower", "two_tower_mappings.pkl")
        if os.path.exists(mappings_path):
            with open(mappings_path, "rb") as f:
                self.two_tower_mappings = pickle_load_compatible(f)
            self.user_embeddings = np.load(os.path.join(self.models_dir, "two_tower", "user_embeddings.npy"))
            self.book_embeddings = np.load(os.path.join(self.models_dir, "two_tower", "book_embeddings.npy"))
            
            # Initialize & load FAISS index
            self.faiss_retriever.load_index()
        else:
            print("Warning: Two-Tower model assets not found. Deep learning & FAISS features will be disabled.")
            
        # Load XGBoost ranker
        self.xgboost_ranker.load()
        print("All sub-models loaded successfully.")
        
    def get_user_history_pos_tfidf(self, user_id):
        # Retrieve books rated highly (>= 4) or liked by user
        user_pos_interactions = self.df_interactions[
            (self.df_interactions["user_id"] == user_id) & 
            ((self.df_interactions["rating"] >= 4) | (self.df_interactions["interaction_type"].isin(["like", "shelve"])))
        ]
        
        b_ids = user_pos_interactions["book_id"].unique()
        indices = [self.content_rec.indices_by_id[b] for b in b_ids if b in self.content_rec.indices_by_id]
        if indices:
            return self.content_rec.tfidf_matrix[indices]
        return None
        
    def recommend(self, user_id, top_n=10, weights=None, filter_rated_book_ids=None):
        if weights is None:
            # Default weights
            weights = {
                "content": 0.20,
                "collaborative": 0.30,
                "two_tower": 0.20,
                "ranking": 0.30
            }
            
        # Verify user exists
        user_rows = self.df_users[self.df_users["user_id"] == user_id]
        if user_rows.empty:
            print(f"User {user_id} not found. Running Cold Start recommend.")
            return self.recommend_cold_start_new_user(user_genres=[], user_authors=[], top_n=top_n)
            
        user_row = user_rows.iloc[0]
        
        # 1. Candidate Retrieval (Stage 1)
        retrieved_candidates = set()
        
        # A. Content-Based Retrieval
        # Find books similar to the user's top-rated books
        user_pos_interactions = self.df_interactions[
            (self.df_interactions["user_id"] == user_id) & (self.df_interactions["rating"] >= 4)
        ]
        if not user_pos_interactions.empty:
            for _, r in user_pos_interactions.head(5).iterrows():
                sim_recs = self.content_rec.recommend_similar_books(book_id=r["book_id"], top_n=10)
                for rec in sim_recs:
                    retrieved_candidates.add(rec["book_id"])
                    
        # B. Collaborative Filtering Retrieval (SVD Predictions)
        # Fetch SVD predicted highest items from catalog
        # Since running predict on all books is slow, we can sample 200 popular books user hasn't rated
        if filter_rated_book_ids is None:
            user_ratings = self.df_interactions[self.df_interactions["user_id"] == user_id]
            rated_book_ids = set(user_ratings["book_id"].unique())
        else:
            rated_book_ids = set(filter_rated_book_ids)
        
        popular_books = self.df_books.sort_values("ratings_count", ascending=False).head(300)["book_id"].tolist()
        unrated_popular = [b for b in popular_books if b not in rated_book_ids]
        
        svd_scores = []
        for b_id in unrated_popular:
            pred = self.collab_rec.predict_rating(user_id, b_id)
            svd_scores.append((b_id, pred))
        svd_scores.sort(key=lambda x: x[1], reverse=True)
        for b_id, _ in svd_scores[:40]:
            retrieved_candidates.add(b_id)
            
        # C. Two-Tower Deep Learning Retrieval (FAISS)
        if self.user_embeddings is not None and self.two_tower_mappings is not None:
            try:
                u_idx = self.two_tower_mappings["user_encoder"].transform([user_id])[0] + 1
                user_emb = self.user_embeddings[u_idx]
                
                faiss_b_ids, faiss_sims = self.faiss_retriever.retrieve_candidates(user_emb, top_k=50)
                for b_id in faiss_b_ids:
                    if b_id not in rated_book_ids:
                        retrieved_candidates.add(b_id)
            except Exception as e:
                print(f"FAISS Retrieval failed for user {user_id}: {e}")
                
        # Clean candidates: remove already rated books
        final_candidates = list(retrieved_candidates - rated_book_ids)
        
        # If we have very few candidates, pad with popular books
        if len(final_candidates) < 20:
            all_popular = self.df_books.sort_values("ratings_count", ascending=False).head(100)["book_id"].tolist()
            for b in all_popular:
                if b not in rated_book_ids:
                    final_candidates.append(b)
            final_candidates = list(set(final_candidates))
            
        # Fetch metadata for candidates
        candidate_books = self.df_books[self.df_books["book_id"].isin(final_candidates)]
        
        # 2. Candidate Ranking (Stage 2) & Feature Extraction
        user_pos_tfidf = self.get_user_history_pos_tfidf(user_id)
        
        # Run XGBoost scoring
        ranked_candidates = self.xgboost_ranker.rank_candidates(
            user_row, candidate_books, self.user_embeddings, self.book_embeddings,
            self.collab_rec, self.content_rec, user_pos_tfidf, top_n=len(candidate_books)
        )
        
        # 3. Hybrid Score Calculation
        hybrid_recs = []
        
        # Extract individual model outputs and normalize to [0,1]
        # XGBoost outputs probabilities [0,1] naturally.
        # SVD returns ratings [1,5], normalize to (pred - 1) / 4
        # Two-tower returns cosine [0,1] or [-1,1], normalize/clip to [0,1]
        # Content returns TF-IDF cosine [0,1]
        for rc in ranked_candidates:
            b_id = rc["book_id"]
            feats = rc["features"]
            xgb_score = rc["score"] # XGBoost prob
            
            content_score = feats["content_similarity"]
            collab_score = (feats["svd_prediction"] - 1.0) / 4.0
            collab_score = max(0.0, min(1.0, collab_score))
            
            two_tower_score = max(0.0, min(1.0, feats["two_tower_similarity"]))
            
            # Hybrid weighted combination
            final_score = (
                weights["content"] * content_score +
                weights["collaborative"] * collab_score +
                weights["two_tower"] * two_tower_score +
                weights["ranking"] * xgb_score
            )
            
            # Generate Explainable AI Reasoning
            reason = self._generate_explanation(user_row, rc)
            
            hybrid_recs.append({
                "book_id": b_id,
                "title": rc["title"],
                "author": rc["author"],
                "genres": rc["genres"],
                "score": round(final_score, 4),
                "reason": reason
            })
            
        # Sort hybrid recommendations by final score descending
        hybrid_recs.sort(key=lambda x: x["score"], reverse=True)
        return hybrid_recs[:top_n]
        
    def recommend_cold_start_new_user(self, user_genres=None, user_authors=None, top_n=10):
        # Handle cases where preferences are empty
        user_genres = [g.lower() for g in user_genres] if user_genres else []
        user_authors = [a.lower() for a in user_authors] if user_authors else []
        
        scores = []
        for idx, row in self.df_books.iterrows():
            book_genres = set(row["standardized_genres"].split("|"))
            book_author = row["author"].lower()
            
            # 1. Genre score (ratio of overlap)
            genre_score = 0.0
            if user_genres:
                overlap = len(book_genres.intersection(user_genres))
                genre_score = overlap / len(user_genres)
                
            # 2. Author score
            author_score = 1.0 if book_author in user_authors else 0.0
            
            # 3. Popularity score (log-normalized average rating and ratings count)
            popularity_score = float(row["average_rating"] / 5.0) * 0.4 + min(1.0, float(row["ratings_count"] / 100000.0)) * 0.6
            
            # Combine
            if user_genres or user_authors:
                final_score = 0.5 * genre_score + 0.3 * author_score + 0.2 * popularity_score
            else:
                # If no preference selected, return popular books
                final_score = popularity_score
                
            # Explanation
            reasons = []
            if genre_score > 0 and user_genres:
                matching = book_genres.intersection(user_genres)
                matching_str = ", ".join([g.capitalize() for g in list(matching)[:2]])
                reasons.append(f"you enjoy similar {matching_str} books")
            if author_score > 0:
                reasons.append(f"it's written by one of your favorite authors: {row['author']}")
            if not reasons:
                reasons.append("it is highly rated and popular among other readers")
                
            reason = "Recommended because " + " and ".join(reasons) + "."
            
            scores.append({
                "book_id": int(row["book_id"]),
                "title": row["title"],
                "author": row["author"],
                "genres": row["genres"],
                "score": round(final_score, 4),
                "reason": reason
            })
            
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_n]
        
    def _generate_explanation(self, user_row, ranked_candidate):
        feats = ranked_candidate["features"]
        title = ranked_candidate["title"]
        author = ranked_candidate["author"]
        
        reasons = []
        
        # 1. Author Match
        if feats["author_match_score"] > 0:
            reasons.append(f"you like books by {author}")
            
        # 2. Genre Match
        if feats["genre_match_score"] > 0.4:
            user_genres = set(user_row["favorite_genres"].split("|"))
            book_genres = set(ranked_candidate["genres"].split("|"))
            overlap = user_genres.intersection(book_genres)
            if overlap:
                reasons.append(f"both you and this book share interests in {', '.join(list(overlap)[:2]).capitalize()}")
                
        # 3. Content Similarity
        if feats["content_similarity"] > 0.5:
            reasons.append("it has descriptive themes and plots similar to books in your library")
            
        # 4. Collaborative Score
        if feats["svd_prediction"] >= 4.0:
            reasons.append("readers with similar reading tastes rated this highly")
            
        # Compose final reason
        if not reasons:
            return "Recommended based on similar user reading trends."
            
        # Return top 2 explanations joined
        return "Recommended because " + " and ".join(reasons[:2]) + "."

def pickle_load_compatible(file_obj):
    # Standard pickle load helper
    return pickle.load(file_obj)
