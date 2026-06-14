import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["axes.titlesize"] = 16

def run_eda():
    print("Running Exploratory Data Analysis (EDA)...")
    
    os.makedirs("reports", exist_ok=True)
    
    # Load processed data
    df_books = pd.read_csv("data/processed_books.csv")
    df_users = pd.read_csv("data/processed_users.csv")
    df_interactions = pd.read_csv("data/interactions.csv")
    
    # Drop rows without rating for rating analysis
    df_ratings = df_interactions.dropna(subset=["rating"])
    
    print(f"Loaded {len(df_books)} books, {len(df_users)} users, and {len(df_interactions)} interactions ({len(df_ratings)} ratings).")
    
    # -------------------------------------------------------------
    # 1. Ratings Distribution Chart
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.countplot(x="rating", data=df_ratings, palette="viridis")
    plt.title("Distribution of User Ratings")
    plt.xlabel("Rating (Stars)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("reports/rating_distribution.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # 2. User Activity Distribution Chart
    # -------------------------------------------------------------
    user_activity = df_interactions.groupby("user_id").size()
    plt.figure(figsize=(10, 5))
    sns.histplot(user_activity, kde=True, color="skyblue", bins=30)
    plt.title("Distribution of User Activity (Interactions per User)")
    plt.xlabel("Number of Interactions")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig("reports/user_activity_distribution.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # 3. Genre Popularity Chart
    # -------------------------------------------------------------
    # Flatten genres
    genres_list = []
    for g_str in df_books["standardized_genres"].dropna():
        genres_list.extend(g_str.split("|"))
        
    df_genres = pd.Series(genres_list).value_counts().reset_index()
    df_genres.columns = ["Genre", "Count"]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Count", y="Genre", data=df_genres, palette="rocket")
    plt.title("Most Common Genres in Catalog")
    plt.xlabel("Number of Books")
    plt.ylabel("Genre")
    plt.tight_layout()
    plt.savefig("reports/genre_distribution.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # 4. Top Authors Chart
    # -------------------------------------------------------------
    df_authors = df_books["author"].value_counts().head(15).reset_index()
    df_authors.columns = ["Author", "Count"]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Count", y="Author", data=df_authors, palette="mako")
    plt.title("Top 15 Authors by Book Count")
    plt.xlabel("Number of Books")
    plt.ylabel("Author")
    plt.tight_layout()
    plt.savefig("reports/author_distribution.png", dpi=150)
    plt.close()
    
    # -------------------------------------------------------------
    # 5. Average Rating per Genre Heatmap/Bar chart
    # -------------------------------------------------------------
    # We join ratings and books to get rating per genre
    df_ratings_joined = df_ratings.merge(df_books, on="book_id")
    
    genre_ratings = []
    for idx, row in df_ratings_joined.iterrows():
        for g in row["standardized_genres"].split("|"):
            genre_ratings.append({"genre": g, "rating": row["rating"]})
            
    df_genre_ratings = pd.DataFrame(genre_ratings)
    avg_genre_ratings = df_genre_ratings.groupby("genre")["rating"].mean().reset_index().sort_values("rating", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="rating", y="genre", data=avg_genre_ratings, palette="crest")
    plt.title("Average Rating by Genre")
    plt.xlabel("Average Rating")
    plt.ylabel("Genre")
    plt.xlim(1.0, 5.0)
    plt.tight_layout()
    plt.savefig("reports/genre_rating_analysis.png", dpi=150)
    plt.close()
    
    # Generate statistics text report
    with open("reports/statistics_report.txt", "w") as f:
        f.write("=== BOOK RECOMMENDATION SYSTEM - DATASET STATISTICS ===\n\n")
        f.write(f"Total Books: {len(df_books)}\n")
        f.write(f"Total Users: {len(df_users)}\n")
        f.write(f"Total Interactions: {len(df_interactions)}\n")
        f.write(f"Total Explicit Ratings: {len(df_ratings)}\n")
        f.write(f"Average System Rating: {df_ratings['rating'].mean():.2f} / 5.0\n\n")
        
        f.write("--- Top 10 Most Popular Books (Interactions count) ---\n")
        pop_books = df_interactions.groupby("book_id").size().sort_values(ascending=False).head(10)
        for rank, (b_id, count) in enumerate(pop_books.items(), 1):
            title = df_books[df_books["book_id"] == b_id]["title"].values[0]
            author = df_books[df_books["book_id"] == b_id]["author"].values[0]
            f.write(f"{rank}. {title} by {author} ({count} interactions)\n")
            
        f.write("\n--- Top 10 Highest Rated Books (with at least 15 ratings) ---\n")
        book_ratings_stats = df_ratings.groupby("book_id").agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        high_rated = book_ratings_stats[book_ratings_stats["rating_count"] >= 15].sort_values("avg_rating", ascending=False).head(10)
        for rank, (b_id, row_stats) in enumerate(high_rated.iterrows(), 1):
            title = df_books[df_books["book_id"] == b_id]["title"].values[0]
            author = df_books[df_books["book_id"] == b_id]["author"].values[0]
            f.write(f"{rank}. {title} by {author} (Avg: {row_stats['avg_rating']:.2f}, Count: {int(row_stats['rating_count'])})\n")
            
        f.write("\n--- Top 10 Most Active Users ---\n")
        active_users = user_activity.sort_values(ascending=False).head(10)
        for rank, (u_id, count) in enumerate(active_users.items(), 1):
            username = df_users[df_users["user_id"] == u_id]["username"].values[0]
            f.write(f"{rank}. {username} (ID: {u_id}) with {count} interactions\n")
            
    print("EDA Complete! Charts and report generated in reports/ folder.")

if __name__ == "__main__":
    run_eda()
