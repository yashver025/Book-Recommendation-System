import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from models.two_tower.model import TwoTowerModel

class RecDataset(Dataset):
    def __init__(self, df_interactions, df_books, df_users, user_encoder, book_encoder, 
                 author_encoder, genre_to_idx, tfidf_vectorizer, num_negatives=4, is_train=True):
        self.df_interactions = df_interactions.copy()
        self.df_books = df_books.copy()
        self.df_users = df_users.copy()
        self.user_encoder = user_encoder
        self.book_encoder = book_encoder
        self.author_encoder = author_encoder
        self.genre_to_idx = genre_to_idx
        self.tfidf_vectorizer = tfidf_vectorizer
        self.num_negatives = num_negatives
        self.is_train = is_train
        
        # Build quick lookups for user and book rows
        self.user_lookup = {row["user_id"]: row for _, row in self.df_users.iterrows()}
        self.book_lookup = {row["book_id"]: row for _, row in self.df_books.iterrows()}
        
        # Get all book IDs for negative sampling
        self.all_book_ids = self.df_books["book_id"].unique()
        
        # Build user-interacted sets for negative sampling
        self.user_interacted = self.df_interactions.groupby("user_id")["book_id"].apply(set).to_dict()
        
        # Create fast lookup dictionaries for encoders to avoid scikit-learn transform overhead
        self.user_to_code = {u: idx for idx, u in enumerate(self.user_encoder.classes_)}
        self.book_to_code = {b: idx for idx, b in enumerate(self.book_encoder.classes_)}
        self.author_to_code = {a: idx for idx, a in enumerate(self.author_encoder.classes_)}
        
        # Precompute book TF-IDF vectors to avoid slow dynamic transform calls in __getitem__
        self.book_tfidfs = {}
        for _, row in self.df_books.iterrows():
            b_id = row["book_id"]
            desc_soup = (
                "genres: " + row["standardized_genres"].replace("|", " ") + " " +
                "author: " + row["author"].lower().replace(" ", "_") + " " +
                "description: " + row["cleaned_description"]
            )
            self.book_tfidfs[b_id] = self.tfidf_vectorizer.transform([desc_soup]).toarray()[0].astype(np.float32)
            
        self.instances = self._prepare_data()
        
    def _prepare_data(self):
        instances = []
        for _, row in self.df_interactions.iterrows():
            u_id = row["user_id"]
            b_id = row["book_id"]
            
            # Positive sample
            instances.append((u_id, b_id, 1.0))
            
            # Negative samples (only during training)
            if self.is_train:
                negatives = []
                interacted = self.user_interacted.get(u_id, set())
                while len(negatives) < self.num_negatives:
                    neg_b_id = np.random.choice(self.all_book_ids)
                    if neg_b_id not in interacted and neg_b_id not in negatives:
                        negatives.append(neg_b_id)
                        
                for neg_b in negatives:
                    instances.append((u_id, neg_b, 0.0))
                    
        return instances
        
    def __len__(self):
        return len(self.instances)
        
    def __getitem__(self, idx):
        u_id, b_id, label = self.instances[idx]
        
        # Fetch user metadata
        u_row = self.user_lookup[u_id]
        # Fetch book metadata
        b_row = self.book_lookup[b_id]
        
        # Encode User ID (add 1 because padding index is 0)
        user_code = self.user_to_code[u_id] + 1
        
        # Encode User Favorite Genres (multi-hot)
        fav_genres_multi = np.zeros(len(self.genre_to_idx), dtype=np.float32)
        if isinstance(u_row["favorite_genres"], str):
            for g in u_row["favorite_genres"].split("|"):
                if g in self.genre_to_idx:
                    fav_genres_multi[self.genre_to_idx[g]] = 1.0
                    
        # Encode User Favorite Authors (multi-hot)
        fav_authors_multi = np.zeros(len(self.author_to_code), dtype=np.float32)
        if isinstance(u_row["favorite_authors"], str):
            for a in u_row["favorite_authors"].split("|"):
                # Clean author key to match encoder
                if a in self.author_to_code:
                    fav_authors_multi[self.author_to_code[a]] = 1.0
                    
        # Encode Book ID
        book_code = self.book_to_code[b_id] + 1
        
        # Encode Book Author
        author_code = 0
        if b_row["author"] in self.author_to_code:
            author_code = self.author_to_code[b_row["author"]] + 1
            
        # Encode Book Genres (multi-hot)
        book_genres_multi = np.zeros(len(self.genre_to_idx), dtype=np.float32)
        if isinstance(b_row["standardized_genres"], str):
            for g in b_row["standardized_genres"].split("|"):
                if g in self.genre_to_idx:
                    book_genres_multi[self.genre_to_idx[g]] = 1.0
                    
        # Get Book Description TF-IDF (precomputed fast lookup)
        desc_tfidf = self.book_tfidfs[b_id]
        
        return {
            "user_id": user_code,
            "fav_genres": fav_genres_multi,
            "fav_authors": fav_authors_multi
        }, {
            "book_id": book_code,
            "author_id": author_code,
            "genres": book_genres_multi,
            "desc_tfidf": desc_tfidf
        }, label

def train_two_tower():
    print("Training PyTorch Two-Tower Recommendation Model...")
    
    # Check paths
    if not (os.path.exists("data/train_interactions.csv") and os.path.exists("data/processed_books.csv") and os.path.exists("data/processed_users.csv")):
        raise FileNotFoundError("Processed datasets missing. Run preprocess.py first.")
        
    df_train = pd.read_csv("data/train_interactions.csv")
    df_val = pd.read_csv("data/val_interactions.csv")
    df_books = pd.read_csv("data/processed_books.csv")
    df_users = pd.read_csv("data/processed_users.csv")
    
    # 1. Fit Label Encoders
    user_encoder = LabelEncoder().fit(df_users["user_id"])
    book_encoder = LabelEncoder().fit(df_books["book_id"])
    
    # For authors, extract unique authors from books and users
    all_authors = set(df_books["author"].unique())
    for a_str in df_users["favorite_authors"].dropna():
        for a in a_str.split("|"):
            all_authors.add(a)
    author_encoder = LabelEncoder().fit(list(all_authors))
    
    # For genres, extract unique genres
    all_genres = set()
    for g_str in df_books["standardized_genres"].dropna():
        for g in g_str.split("|"):
            all_genres.add(g)
    genre_to_idx = {g: idx for idx, g in enumerate(sorted(all_genres))}
    
    # Fit TF-IDF Vectorizer on book descriptions/metadata
    df_books["metadata_soup"] = (
        "genres: " + df_books["standardized_genres"].str.replace("|", " ") + " " +
        "author: " + df_books["author"].str.lower().str.replace(" ", "_") + " " +
        "description: " + df_books["cleaned_description"]
    )
    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_vectorizer.fit(df_books["metadata_soup"])
    
    # Create dataset objects
    train_dataset = RecDataset(
        df_train, df_books, df_users, user_encoder, book_encoder, 
        author_encoder, genre_to_idx, tfidf_vectorizer, num_negatives=4, is_train=True
    )
    val_dataset = RecDataset(
        df_val, df_books, df_users, user_encoder, book_encoder, 
        author_encoder, genre_to_idx, tfidf_vectorizer, num_negatives=0, is_train=False
    )
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    # Model dimensions
    num_users = len(user_encoder.classes_)
    num_books = len(book_encoder.classes_)
    num_genres = len(genre_to_idx)
    num_authors = len(author_encoder.classes_)
    tfidf_dim = len(tfidf_vectorizer.vocabulary_)
    
    # Initialize Two-Tower Model
    model = TwoTowerModel(
        num_users=num_users,
        num_books=num_books,
        num_genres=num_genres,
        num_authors=num_authors,
        tfidf_dim=tfidf_dim,
        output_dim=128
    )
    
    # Loss, Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training Loop
    epochs = 10
    temperature = 10.0 # Scaling factor for cosine similarity
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Training on device: {device}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for user_inputs, book_inputs, labels in train_loader:
            # Move inputs to device
            user_inputs = {k: v.to(device) for k, v in user_inputs.items()}
            book_inputs = {k: v.to(device) for k, v in book_inputs.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            user_emb, book_emb = model(user_inputs, book_inputs)
            
            # Similarity score (dot product)
            # Both are L2-normalized, so dot product = cosine similarity
            sim = torch.sum(user_emb * book_emb, dim=1) # [batch_size]
            logits = sim * temperature
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            
        train_loss /= len(train_dataset)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for user_inputs, book_inputs, labels in val_loader:
                user_inputs = {k: v.to(device) for k, v in user_inputs.items()}
                book_inputs = {k: v.to(device) for k, v in book_inputs.items()}
                labels = labels.to(device)
                
                user_emb, book_emb = model(user_inputs, book_inputs)
                sim = torch.sum(user_emb * book_emb, dim=1)
                logits = sim * temperature
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                
        val_loss /= len(val_dataset)
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
    # Save the model
    os.makedirs("models/two_tower", exist_ok=True)
    torch.save(model.state_dict(), "models/two_tower/two_tower_model.pth")
    
    # Save mappings
    mappings = {
        "user_encoder": user_encoder,
        "book_encoder": book_encoder,
        "author_encoder": author_encoder,
        "genre_to_idx": genre_to_idx,
        "tfidf_vectorizer": tfidf_vectorizer,
        "num_users": num_users,
        "num_books": num_books,
        "num_genres": num_genres,
        "num_authors": num_authors
    }
    with open("models/two_tower/two_tower_mappings.pkl", "wb") as f:
        pickle.dump(mappings, f)
        
    print("Model and mappings saved successfully.")
    
    # Generate and save Embeddings
    print("Generating embedding matrix for all users and books...")
    model.eval()
    
    # Book Embeddings
    # We will construct a dummy dataset containing all books in catalog
    book_embeddings = np.zeros((num_books + 1, 128), dtype=np.float32)
    
    with torch.no_grad():
        for _, row in df_books.iterrows():
            b_id = row["book_id"]
            b_code = book_encoder.transform([b_id])[0] + 1
            
            author_code = 0
            if row["author"] in author_encoder.classes_:
                author_code = author_encoder.transform([row["author"]])[0] + 1
                
            book_genres_multi = np.zeros(num_genres, dtype=np.float32)
            if isinstance(row["standardized_genres"], str):
                for g in row["standardized_genres"].split("|"):
                    if g in genre_to_idx:
                        book_genres_multi[genre_to_idx[g]] = 1.0
                        
            desc_soup = (
                "genres: " + row["standardized_genres"].replace("|", " ") + " " +
                "author: " + row["author"].lower().replace(" ", "_") + " " +
                "description: " + row["cleaned_description"]
            )
            desc_tfidf = tfidf_vectorizer.transform([desc_soup]).toarray()[0].astype(np.float32)
            
            b_id_t = torch.tensor([b_code], dtype=torch.long).to(device)
            a_id_t = torch.tensor([author_code], dtype=torch.long).to(device)
            g_multi_t = torch.tensor([book_genres_multi], dtype=torch.float32).to(device)
            desc_tfidf_t = torch.tensor([desc_tfidf], dtype=torch.float32).to(device)
            
            b_emb = model.book_tower(b_id_t, a_id_t, g_multi_t, desc_tfidf_t)
            book_embeddings[b_code] = b_emb.cpu().numpy()[0]
            
    # User Embeddings
    user_embeddings = np.zeros((num_users + 1, 128), dtype=np.float32)
    
    with torch.no_grad():
        for _, row in df_users.iterrows():
            u_id = row["user_id"]
            u_code = user_encoder.transform([u_id])[0] + 1
            
            fav_genres_multi = np.zeros(num_genres, dtype=np.float32)
            if isinstance(row["favorite_genres"], str):
                for g in row["favorite_genres"].split("|"):
                    if g in genre_to_idx:
                        fav_genres_multi[genre_to_idx[g]] = 1.0
                        
            fav_authors_multi = np.zeros(num_authors, dtype=np.float32)
            if isinstance(row["favorite_authors"], str):
                for a in row["favorite_authors"].split("|"):
                    if a in author_encoder.classes_:
                        fav_authors_multi[author_encoder.transform([a])[0]] = 1.0
                        
            u_id_t = torch.tensor([u_code], dtype=torch.long).to(device)
            g_multi_t = torch.tensor([fav_genres_multi], dtype=torch.float32).to(device)
            a_multi_t = torch.tensor([fav_authors_multi], dtype=torch.float32).to(device)
            
            u_emb = model.user_tower(u_id_t, g_multi_t, a_multi_t)
            user_embeddings[u_code] = u_emb.cpu().numpy()[0]
            
    np.save("models/two_tower/book_embeddings.npy", book_embeddings)
    np.save("models/two_tower/user_embeddings.npy", user_embeddings)
    print("User and book embeddings generated and saved.")

if __name__ == "__main__":
    train_two_tower()
