import os
import sys
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.two_tower.train import RecDataset

def test_speed():
    print("Testing Two-Tower Dataset speed...")
    df_train = pd.read_csv("data/train_interactions.csv")
    df_books = pd.read_csv("data/processed_books.csv")
    df_users = pd.read_csv("data/processed_users.csv")
    
    user_encoder = LabelEncoder().fit(df_users["user_id"])
    book_encoder = LabelEncoder().fit(df_books["book_id"])
    
    all_authors = set(df_books["author"].unique())
    for a_str in df_users["favorite_authors"].dropna():
        for a in a_str.split("|"):
            all_authors.add(a)
    author_encoder = LabelEncoder().fit(list(all_authors))
    
    all_genres = set()
    for g_str in df_books["standardized_genres"].dropna():
        for g in g_str.split("|"):
            all_genres.add(g)
    genre_to_idx = {g: idx for idx, g in enumerate(sorted(all_genres))}
    
    df_books["metadata_soup"] = (
        "genres: " + df_books["standardized_genres"].str.replace("|", " ") + " " +
        "author: " + df_books["author"].str.lower().str.replace(" ", "_") + " " +
        "description: " + df_books["cleaned_description"]
    )
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_vectorizer.fit(df_books["metadata_soup"])
    
    t0 = time.time()
    train_dataset = RecDataset(
        df_train, df_books, df_users, user_encoder, book_encoder, 
        author_encoder, genre_to_idx, tfidf_vectorizer, num_negatives=4, is_train=True
    )
    print(f"Dataset initialization & precomputation took: {time.time() - t0:.4f} seconds.")
    print(f"Total instances in training set: {len(train_dataset)}")
    
    # Measure __getitem__ speed
    t0 = time.time()
    num_samples = 5000
    for i in range(num_samples):
        _ = train_dataset[i]
    dt = time.time() - t0
    print(f"Reading {num_samples} items dynamically took: {dt:.4f} seconds ({dt/num_samples*1000:.4f} ms per sample).")
    
    # Test DataLoader iteration
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False)
    t0 = time.time()
    batch_count = 0
    for batch in train_loader:
        batch_count += 1
        if batch_count >= 10:
            break
    dt = time.time() - t0
    print(f"Loading 10 batches (size 256) took: {dt:.4f} seconds ({dt/10*1000:.4f} ms per batch).")

if __name__ == "__main__":
    test_speed()
