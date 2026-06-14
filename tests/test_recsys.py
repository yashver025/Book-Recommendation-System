import os
import sys
import unittest
import numpy as np
import pandas as pd
import torch

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.content_based.recommender import ContentBasedRecommender
from models.collaborative.collaborative import CollaborativeRecommender
from models.two_tower.model import TwoTowerModel
from models.ranking.ranking import XGBoostRanker
from models.hybrid import HybridRecommendationEngine
from database.db import SessionLocal
from database.models import Book, User, Rating

class TestBookRecSys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Paths
        cls.data_dir = "data"
        cls.models_dir = "models"
        
        # Verify files exist before running tests
        cls.has_data = os.path.exists(os.path.join(cls.data_dir, "processed_books.csv"))
        if cls.has_data:
            cls.df_books = pd.read_csv(os.path.join(cls.data_dir, "processed_books.csv"))
            cls.df_users = pd.read_csv(os.path.join(cls.data_dir, "processed_users.csv"))
            cls.df_interactions = pd.read_csv(os.path.join(cls.data_dir, "interactions.csv"))
            cls.df_train = pd.read_csv(os.path.join(cls.data_dir, "train_interactions.csv"))
            cls.df_val = pd.read_csv(os.path.join(cls.data_dir, "val_interactions.csv"))
            
    def test_01_data_exists(self):
        self.assertTrue(self.has_data, "Processed CSV datasets missing. Run data generator/preprocessing first.")
        self.assertTrue(len(self.df_books) > 0, "Book catalog is empty.")
        self.assertTrue(len(self.df_users) > 0, "Users dataset is empty.")
        self.assertTrue(len(self.df_interactions) > 0, "Interactions dataset is empty.")
        
    def test_02_content_based_recommender(self):
        if not self.has_data:
            self.skipTest("No data available.")
            
        recommender = ContentBasedRecommender()
        recommender.fit(self.df_books)
        
        # Test similar books lookup
        test_title = self.df_books.iloc[0]["title"]
        recs = recommender.recommend_similar_books(book_title=test_title, top_n=5)
        
        self.assertEqual(len(recs), 5, "Content-based recommendations count mismatch.")
        self.assertTrue(all("book_id" in r for r in recs), "book_id missing from recommendation items.")
        self.assertTrue(all("score" in r for r in recs), "score missing from recommendation items.")
        self.assertTrue(all("reason" in r for r in recs), "reason missing from recommendation items.")
        
    def test_03_collaborative_recommender(self):
        if not self.has_data:
            self.skipTest("No data available.")
            
        recommender = CollaborativeRecommender()
        recommender.fit(self.df_train)
        
        # Test rating prediction
        test_user = self.df_train.iloc[0]["user_id"]
        test_book = self.df_train.iloc[0]["book_id"]
        
        pred_svd = recommender.predict_rating(test_user, test_book, model_type="svd")
        pred_knn = recommender.predict_rating(test_user, test_book, model_type="knn")
        
        self.assertTrue(1.0 <= pred_svd <= 5.0, f"SVD prediction {pred_svd} out of range [1,5].")
        self.assertTrue(1.0 <= pred_knn <= 5.0, f"KNN prediction {pred_knn} out of range [1,5].")
        
    def test_04_two_tower_dimensions(self):
        # Test model tensor dimensions
        num_users = 100
        num_books = 200
        num_genres = 10
        num_authors = 15
        tfidf_dim = 1000
        embedding_dim = 32
        output_dim = 128
        
        model = TwoTowerModel(
            num_users=num_users,
            num_books=num_books,
            num_genres=num_genres,
            num_authors=num_authors,
            tfidf_dim=tfidf_dim,
            embedding_dim=embedding_dim,
            output_dim=output_dim
        )
        
        # Dummy inputs
        batch_size = 4
        user_inputs = {
            "user_id": torch.randint(1, num_users, (batch_size,), dtype=torch.long),
            "fav_genres": torch.rand((batch_size, num_genres), dtype=torch.float32),
            "fav_authors": torch.rand((batch_size, num_authors), dtype=torch.float32)
        }
        book_inputs = {
            "book_id": torch.randint(1, num_books, (batch_size,), dtype=torch.long),
            "author_id": torch.randint(1, num_authors, (batch_size,), dtype=torch.long),
            "genres": torch.rand((batch_size, num_genres), dtype=torch.float32),
            "desc_tfidf": torch.rand((batch_size, tfidf_dim), dtype=torch.float32)
        }
        
        u_emb, b_emb = model(user_inputs, book_inputs)
        
        self.assertEqual(u_emb.shape, (batch_size, output_dim), "User tower output embedding shape mismatch.")
        self.assertEqual(b_emb.shape, (batch_size, output_dim), "Book tower output embedding shape mismatch.")
        
        # Test L2 normalization
        self.assertTrue(torch.allclose(torch.norm(u_emb, p=2, dim=1), torch.ones(batch_size)), "User embeddings not L2 normalized.")
        self.assertTrue(torch.allclose(torch.norm(b_emb, p=2, dim=1), torch.ones(batch_size)), "Book embeddings not L2 normalized.")
        
    def test_05_database_connections(self):
        try:
            db = SessionLocal()
            # Try fetching a book
            book = db.query(Book).first()
            user = db.query(User).first()
            rating = db.query(Rating).first()
            db.close()
            
            # If seeding ran, these should not be None
            self.assertIsNotNone(book, "Books table is empty in Database.")
            self.assertIsNotNone(user, "Users table is empty in Database.")
            self.assertIsNotNone(rating, "Ratings table is empty in Database.")
        except Exception as e:
            self.fail(f"Database connection or query failed: {e}")

if __name__ == "__main__":
    unittest.main()
