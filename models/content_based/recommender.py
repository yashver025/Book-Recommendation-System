import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self, model_dir="models/content_based"):
        self.model_dir = model_dir
        self.vectorizer = None
        self.tfidf_matrix = None
        self.df_books = None
        self.indices_by_title = {}
        self.indices_by_id = {}
        os.makedirs(self.model_dir, exist_ok=True)
        
    def fit(self, df_books):
        print("Fitting TF-IDF Content-Based Recommender...")
        self.df_books = df_books.copy()
        
        # Combine metadata into a "soup" of features
        # We give more weight to genres and authors by repeating them
        self.df_books["metadata_soup"] = (
            "genres: " + self.df_books["standardized_genres"].str.replace("|", " ") + " " +
            "genres: " + self.df_books["standardized_genres"].str.replace("|", " ") + " " + # double weight
            "author: " + self.df_books["author"].str.lower().str.replace(" ", "_") + " " +
            "author: " + self.df_books["author"].str.lower().str.replace(" ", "_") + " " + # double weight
            "description: " + self.df_books["cleaned_description"]
        )
        
        # Train TfidfVectorizer
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df_books["metadata_soup"])
        
        # Build index maps
        self.indices_by_title = {row["title"].lower(): idx for idx, row in self.df_books.iterrows()}
        self.indices_by_id = {row["book_id"]: idx for idx, row in self.df_books.iterrows()}
        
        print("Fitting complete.")
        
    def save(self):
        print("Saving Content-Based model artifacts...")
        with open(os.path.join(self.model_dir, "vectorizer.pkl"), "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(self.model_dir, "tfidf_matrix.pkl"), "wb") as f:
            pickle.dump(self.tfidf_matrix, f)
        with open(os.path.join(self.model_dir, "books_metadata.pkl"), "wb") as f:
            pickle.dump(self.df_books, f)
        print("Saved successfully.")
        
    def load(self):
        print("Loading Content-Based model artifacts...")
        vectorizer_path = os.path.join(self.model_dir, "vectorizer.pkl")
        matrix_path = os.path.join(self.model_dir, "tfidf_matrix.pkl")
        books_path = os.path.join(self.model_dir, "books_metadata.pkl")
        
        if not (os.path.exists(vectorizer_path) and os.path.exists(matrix_path) and os.path.exists(books_path)):
            raise FileNotFoundError("Model artifacts not found. Fit the model first.")
            
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(matrix_path, "rb") as f:
            self.tfidf_matrix = pickle.load(f)
        with open(books_path, "rb") as f:
            self.df_books = pickle.load(f)
            
        # Rebuild index maps
        self.indices_by_title = {row["title"].lower(): idx for idx, row in self.df_books.iterrows()}
        self.indices_by_id = {row["book_id"]: idx for idx, row in self.df_books.iterrows()}
        print("Loaded successfully.")
        
    def recommend_similar_books(self, book_title=None, book_id=None, top_n=10):
        # Resolve the index
        idx = None
        if book_id is not None:
            idx = self.indices_by_id.get(book_id)
        elif book_title is not None:
            idx = self.indices_by_title.get(book_title.lower())
            
        if idx is None:
            print(f"Book '{book_title or book_id}' not found in catalog.")
            return []
            
        target_book = self.df_books.iloc[idx]
        
        # Calculate cosine similarity of target book with all books
        cosine_sim = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        
        # Sort indices
        similar_indices = np.argsort(cosine_sim)[::-1]
        
        # Filter out the target book itself
        similar_indices = [i for i in similar_indices if i != idx]
        
        recommendations = []
        for i in similar_indices[:top_n]:
            sim_book = self.df_books.iloc[i]
            sim_score = cosine_sim[i]
            
            # Generate reasoning explanation
            overlap_genres = set(target_book["standardized_genres"].split("|")).intersection(
                set(sim_book["standardized_genres"].split("|"))
            )
            same_author = target_book["author"] == sim_book["author"]
            
            reason = "Recommended because "
            reasons = []
            if same_author:
                reasons.append(f"both are written by {target_book['author']}")
            if overlap_genres:
                genres_formatted = ", ".join([g.capitalize() for g in list(overlap_genres)[:3]])
                reasons.append(f"both belong to the {genres_formatted} genres")
            
            # Check description keywords overlap
            kw_target = set(target_book["cleaned_description"].split())
            kw_sim = set(sim_book["cleaned_description"].split())
            common_kw = kw_target.intersection(kw_sim) - {"and", "the", "a", "of", "in", "to", "is", "about", "for", "with", "this", "by", "that", "an"}
            if len(common_kw) >= 2:
                reasons.append(f"they share key concepts like: {', '.join(list(common_kw)[:2])}")
                
            if reasons:
                reason += " and ".join(reasons) + "."
            else:
                reason += "they have similar thematic profiles."
                
            recommendations.append({
                "book_id": int(sim_book["book_id"]),
                "title": sim_book["title"],
                "author": sim_book["author"],
                "genres": sim_book["genres"],
                "score": float(sim_score),
                "reason": reason
            })
            
        return recommendations

if __name__ == "__main__":
    # Test content recommender locally
    if os.path.exists("data/processed_books.csv"):
        df_books = pd.read_csv("data/processed_books.csv")
        recommender = ContentBasedRecommender()
        recommender.fit(df_books)
        recommender.save()
        
        recommender.load()
        # Recommend based on first book in catalog
        test_title = df_books.iloc[0]["title"]
        print(f"\nRecommendations for '{test_title}':")
        recs = recommender.recommend_similar_books(book_title=test_title, top_n=3)
        for r in recs:
            print(f"- {r['title']} by {r['author']} (Score: {r['score']:.4f})")
            print(f"  Reason: {r['reason']}")
