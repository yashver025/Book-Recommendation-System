import os
import re
import pandas as pd
import numpy as np

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove HTML tags, special chars, extra whitespaces
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^\w\s\-\.,!?']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def preprocess_data():
    print("Preprocessing raw datasets...")
    
    # Check if raw files exist. If not, raise error.
    if not os.path.exists("data/raw_books.csv") or not os.path.exists("data/raw_users.csv") or not os.path.exists("data/raw_ratings.csv"):
        raise FileNotFoundError("Raw CSV files not found. Run data_generator.py first.")
        
    df_books = pd.read_csv("data/raw_books.csv")
    df_users = pd.read_csv("data/raw_users.csv")
    df_ratings = pd.read_csv("data/raw_ratings.csv")
    
    # 1. Deduplicate
    # Books
    df_books = df_books.drop_duplicates(subset=["book_id"])
    # Users
    df_users = df_users.drop_duplicates(subset=["user_id"])
    # Ratings/Interactions: Keep the latest interaction per user-book pair
    df_ratings["timestamp"] = pd.to_datetime(df_ratings["timestamp"])
    df_ratings = df_ratings.sort_values("timestamp")
    df_ratings = df_ratings.drop_duplicates(subset=["user_id", "book_id"], keep="last")
    
    # 2. Handle missing values & Clean Text
    df_books["description"] = df_books["description"].fillna("")
    df_books["cleaned_description"] = df_books["description"].apply(clean_text)
    
    # Fill average ratings with global average if missing
    global_avg_rating = df_books["average_rating"].mean()
    df_books["average_rating"] = df_books["average_rating"].fillna(global_avg_rating)
    df_books["ratings_count"] = df_books["ratings_count"].fillna(0).astype(int)
    df_books["reviews_count"] = df_books["reviews_count"].fillna(0).astype(int)
    
    # Standardize authors
    df_books["author"] = df_books["author"].fillna("Unknown").str.strip()
    
    # Standardize Genres list (lowercase & sorted)
    def clean_genres(genre_str):
        if not isinstance(genre_str, str):
            return "unknown"
        gs = [g.strip().lower() for g in genre_str.split("|")]
        return "|".join(sorted(gs))
        
    df_books["standardized_genres"] = df_books["genres"].apply(clean_genres)
    
    # Clean users favorite genres/authors
    df_users["favorite_genres"] = df_users["favorite_genres"].fillna("").apply(clean_genres)
    df_users["favorite_authors"] = df_users["favorite_authors"].fillna("").apply(
        lambda x: "|".join(sorted([a.strip().lower() for a in x.split("|")])) if isinstance(x, str) else ""
    )
    
    # 3. Create Interactions split
    # For collaborative filtering, SVD, SVD++ etc. we need user rating matrix
    # Clean ratings: convert NaNs in ratings to 0 or drop? 
    # For ranking model, we need all interactions (implicit + explicit). 
    # Let's save interactions.csv containing all interactions.
    df_ratings.to_csv("data/interactions.csv", index=False)
    
    # Splitting Train / Val / Test (Stratified per User)
    # We want to ensure every user has at least some ratings in train.
    train_list = []
    val_list = []
    test_list = []
    
    for user_id, group in df_ratings.groupby("user_id"):
        n_interactions = len(group)
        if n_interactions >= 5:
            # Shuffle group
            group_shuffled = group.sample(frac=1, random_state=42)
            n_train = int(n_interactions * 0.8)
            n_val = int(n_interactions * 0.1)
            
            train_list.append(group_shuffled.iloc[:n_train])
            val_list.append(group_shuffled.iloc[n_train:n_train+n_val])
            test_list.append(group_shuffled.iloc[n_train+n_val:])
        else:
            # Put all in train if user has very few interactions
            train_list.append(group)
            
    df_train = pd.concat(train_list).sort_values("timestamp")
    df_val = pd.concat(val_list).sort_values("timestamp") if val_list else pd.DataFrame(columns=df_ratings.columns)
    df_test = pd.concat(test_list).sort_values("timestamp") if test_list else pd.DataFrame(columns=df_ratings.columns)
    
    # Save splits
    df_train.to_csv("data/train_interactions.csv", index=False)
    df_val.to_csv("data/val_interactions.csv", index=False)
    df_test.to_csv("data/test_interactions.csv", index=False)
    
    # Save processed users and books
    df_books.to_csv("data/processed_books.csv", index=False)
    df_users.to_csv("data/processed_users.csv", index=False)
    
    print(f"Data Preprocessing Complete!")
    print(f"Processed books: {len(df_books)}")
    print(f"Processed users: {len(df_users)}")
    print(f"Total interactions: {len(df_ratings)}")
    print(f"Train split: {len(df_train)}")
    print(f"Val split: {len(df_val)}")
    print(f"Test split: {len(df_test)}")

if __name__ == "__main__":
    preprocess_data()
