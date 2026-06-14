import os
import random
import pandas as pd
import numpy as np

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Define constants
NUM_BOOKS = 1200
NUM_USERS = 400
NUM_RATINGS = 25000

GENRES = [
    "Fantasy", "Science Fiction", "Mystery", "Thriller", "Romance", 
    "Historical Fiction", "Non-fiction", "Biography", "Self-Help", "Business"
]

AUTHORS = [
    "J.R.R. Tolkien", "George R.R. Martin", "Isaac Asimov", "Arthur C. Clarke",
    "Agatha Christie", "Stephen King", "Gillian Flynn", "Dan Brown",
    "Jane Austen", "Nicholas Sparks", "Colleen Hoover", "Ken Follett",
    "Yuval Noah Harari", "Malcolm Gladwell", "Walter Isaacson", "James Clear",
    "Simon Sinek", "Dale Carnegie", "Peter Thiel", "Sheryl Sandberg",
    "J.K. Rowling", "Brandon Sanderson", "Philip K. Dick", "Neil Gaiman",
    "Ray Bradbury", "Arthur Conan Doyle", "Robert Kiyosaki", "Tim Ferriss",
    "Daniel Kahneman", "Adam Grant"
]

BOOK_TEMPLATES = {
    "Fantasy": [
        "The Chronicles of {}", "The Shadow of {}", "The Heir of {}", 
        "{} and the Lost Kingdom", "The Sword of {}", "Whispers from the {}"
    ],
    "Science Fiction": [
        "The {} Odyssey", "Echoes of {}", "The {} Protocol", 
        "Beyond the {} Frontier", "{} Millennium", "The {} Matrix"
    ],
    "Mystery": [
        "The Murder at {}", "The Case of the {}", "The {} Enigma",
        "Death on {}", "The {} Alibi", "Vanished in {}"
    ],
    "Thriller": [
        "The {} Conspiracy", "{} Point", "The {} Target",
        "Calculated {}", "No Way {}", "The {} Threat"
    ],
    "Romance": [
        "A Love in {}", "Hearts in {}", "The {} Promise",
        "Whispering {}", "{} Forever", "Summer in {}"
    ],
    "Historical Fiction": [
        "The {} of War", "Letters from {}", "The {} Tailor",
        "Legacy of {}", "The {} Secret", "Shadows of {}"
    ],
    "Non-fiction": [
        "The {} of Everything", "Understanding {}", "The {} Era",
        "Decoding {}", "Inside {}", "The {} Perspective"
    ],
    "Biography": [
        "The Life of {}", "{} and His Times", "The {} Legacy",
        "The Story of {}", "Becoming {}", "{} Unveiled"
    ],
    "Self-Help": [
        "Mastering your {}", "The {} Mindset", "Atomic {}",
        "12 Rules of {}", "The Power of {}", "{} Habits"
    ],
    "Business": [
        "The {} Startup", "Leading the {}", "The {} Advantage",
        "Zero to {}", "The {} Economy", "How to Build a {}"
    ]
}

KEYWORD_BANK = {
    "Fantasy": ["magic", "kingdom", "dragon", "quest", "sword", "wizard", "prophecy", "realm", "legendary", "elf", "spell", "hero"],
    "Science Fiction": ["space", "futuristic", "galaxy", "robot", "artificial intelligence", "alien", "technology", "spaceships", "cybernetic", "future", "star", "dimensions"],
    "Mystery": ["detective", "murder", "clue", "suspect", "mystery", "crime", "investigation", "case", "secrets", "puzzle", "truth", "unravel"],
    "Thriller": ["conspiracy", "escape", "danger", "deadly", "assassin", "agent", "chase", "urgency", "survival", "threat", "tension", "action"],
    "Romance": ["love", "passion", "heart", "relationship", "romance", "affection", "emotional", "spark", "lovers", "devotion", "marriage", "date"],
    "Historical Fiction": ["war", "history", "ancient", "era", "century", "empire", "generation", "vintage", "heritage", "past", "rebel", "struggle"],
    "Non-fiction": ["science", "society", "human", "culture", "world", "facts", "analysis", "system", "nature", "exploration", "reality", "mind"],
    "Biography": ["autobiography", "memoir", "legacy", "life", "career", "journey", "personal", "triumph", "struggle", "inspiration", "achievement", "portrait"],
    "Self-Help": ["mindset", "habits", "success", "productivity", "growth", "motivation", "happiness", "discipline", "focus", "goals", "lifestyle", "wisdom"],
    "Business": ["startup", "marketing", "management", "leadership", "finance", "strategy", "economy", "entrepreneur", "innovation", "corporate", "sales", "wealth"]
}

def generate_book_descriptions(genres, title, author):
    # Select keywords from the corresponding genres
    selected_kws = []
    for g in genres:
        selected_kws.extend(random.sample(KEYWORD_BANK[g], k=min(3, len(KEYWORD_BANK[g]))))
    
    # Remove duplicates
    selected_kws = list(set(selected_kws))
    
    desc_templates = [
        f"A compelling read about {', '.join(selected_kws[:-1])} and {selected_kws[-1]}. This book explores deep themes, masterfully written by {author}. It stands out in its representation of {', '.join(genres)}.",
        f"Written by the acclaimed author {author}, this title delivers a gripping narrative centering on {selected_kws[0]} and {selected_kws[1]}. Ideal for fans of {', '.join(genres)}, it offers an immersive journey filled with {', '.join(selected_kws[2:])}.",
        f"Uncovering the intricacies of {selected_kws[0]}, this masterpiece by {author} combines elements of {', '.join(genres)} to create a breathtaking experience. Dive into a world where {selected_kws[1]} and {selected_kws[-1]} shape the ultimate destiny."
    ]
    return random.choice(desc_templates)

def main():
    print("Generating raw synthetic Goodreads dataset...")
    
    os.makedirs("data", exist_ok=True)
    
    # 1. Generate Books
    books_data = []
    for book_id in range(1, NUM_BOOKS + 1):
        # Sample 1-3 genres
        num_g = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        genres = random.sample(GENRES, k=num_g)
        
        # Select author
        author = random.choice(AUTHORS)
        
        # Generate Title based on primary genre
        primary_genre = genres[0]
        template = random.choice(BOOK_TEMPLATES[primary_genre])
        
        # Pick a noun word or random keyword for title formatting
        title_word = random.choice(KEYWORD_BANK[primary_genre]).capitalize()
        title = template.format(title_word)
        
        # Generate description
        description = generate_book_descriptions(genres, title, author)
        
        # Rating distributions
        avg_rating = round(np.random.normal(loc=3.8, scale=0.45), 2)
        avg_rating = max(1.0, min(5.0, avg_rating))
        
        ratings_count = int(np.random.lognormal(mean=6.5, sigma=1.2))
        reviews_count = int(ratings_count * np.random.uniform(0.05, 0.15))
        
        books_data.append({
            "book_id": book_id,
            "title": title,
            "author": author,
            "genres": "|".join(genres),
            "description": description,
            "average_rating": avg_rating,
            "ratings_count": ratings_count,
            "reviews_count": reviews_count
        })
        
    df_books = pd.DataFrame(books_data)
    df_books.to_csv("data/raw_books.csv", index=False)
    print(f"Generated {NUM_BOOKS} books saved to data/raw_books.csv")
    
    # 2. Generate Users
    users_data = []
    for user_id in range(1, NUM_USERS + 1):
        num_fav_genres = random.choice([1, 2, 3])
        fav_genres = random.sample(GENRES, k=num_fav_genres)
        
        num_fav_authors = random.choice([1, 2])
        fav_authors = random.sample(AUTHORS, k=num_fav_authors)
        
        users_data.append({
            "user_id": user_id,
            "username": f"User_{user_id}",
            "favorite_genres": "|".join(fav_genres),
            "favorite_authors": "|".join(fav_authors)
        })
        
    df_users = pd.DataFrame(users_data)
    df_users.to_csv("data/raw_users.csv", index=False)
    print(f"Generated {NUM_USERS} users saved to data/raw_users.csv")
    
    # 3. Generate Ratings and Interactions
    # To build a realistic dataset, users must interact with items they like
    # and rate them according to their preferences.
    ratings_data = []
    
    # Convert list representations for faster lookup
    users_prefs = {
        row["user_id"]: {
            "genres": set(row["favorite_genres"].split("|")),
            "authors": set(row["favorite_authors"].split("|"))
        }
        for _, row in df_users.iterrows()
    }
    
    books_meta = [
        {
            "book_id": row["book_id"],
            "genres": set(row["genres"].split("|")),
            "author": row["author"],
            "avg_rating": row["average_rating"]
        }
        for _, row in df_books.iterrows()
    ]
    
    # For each user, we will assign a probability weight to each book
    # and sample NUM_RATINGS / NUM_USERS books.
    books_per_user = int(NUM_RATINGS / NUM_USERS)
    
    interaction_choices = ["view", "shelve", "rate"]
    interaction_weights = [0.2, 0.1, 0.7] # 70% explicit ratings, 30% implicit interactions
    
    review_templates_pos = [
        "An absolute masterpiece! Loved every page.",
        "Beautifully written and highly engaging.",
        "Strong characters and excellent plot. Highly recommended.",
        "Could not put it down! A fantastic book.",
        "A wonderful read, really connected with the story."
    ]
    
    review_templates_neg = [
        "Not my cup of tea. Found it a bit slow.",
        "Disappointing, execution fell short of the premise.",
        "Very basic characters and predictable plot.",
        "Hard to get through. Not recommended.",
        "Decent idea but poorly written."
    ]
    
    for user_id, prefs in users_prefs.items():
        user_fav_genres = prefs["genres"]
        user_fav_authors = prefs["authors"]
        
        # Calculate affinity weights for all books
        weights = []
        for book in books_meta:
            # Overlap in genres
            genre_overlap = len(book["genres"].intersection(user_fav_genres))
            genre_score = genre_overlap / max(1, len(user_fav_genres))
            
            # Author match
            author_score = 1.0 if book["author"] in user_fav_authors else 0.0
            
            # Rating affinity
            rating_score = (book["avg_rating"] - 1.0) / 4.0
            
            # Final affinity score (determines probability of selection)
            affinity = 0.5 * genre_score + 0.3 * author_score + 0.2 * rating_score
            # Add a small base probability so users occasionally read outside their favorite sphere
            weights.append(affinity + 0.05)
            
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Sample books for this user
        sampled_book_indices = np.random.choice(
            len(books_meta), size=books_per_user, replace=False, p=weights
        )
        
        for idx in sampled_book_indices:
            book = books_meta[idx]
            
            # Calculate actual rating (explicit signal)
            genre_overlap = len(book["genres"].intersection(user_fav_genres))
            genre_score = genre_overlap / max(1, len(user_fav_genres))
            author_score = 1.0 if book["author"] in user_fav_authors else 0.0
            
            # Latent affinity determines expected rating
            expected_rating = 2.5 + 1.5 * genre_score + 0.8 * author_score + 0.2 * (book["avg_rating"] - 3.0)
            # Add noise
            actual_rating = int(np.clip(round(expected_rating + np.random.normal(0, 0.6)), 1, 5))
            
            # Determine interaction type
            itype = np.random.choice(interaction_choices, p=interaction_weights)
            
            # Generate review text for ratings
            review_text = ""
            if itype == "rate":
                if actual_rating >= 4:
                    review_text = random.choice(review_templates_pos)
                elif actual_rating <= 2:
                    review_text = random.choice(review_templates_neg)
                else:
                    review_text = "It was an okay read. Met expectations."
                    
            ratings_data.append({
                "user_id": user_id,
                "book_id": book["book_id"],
                "rating": actual_rating if itype == "rate" else np.nan,
                "interaction_type": itype,
                "review_text": review_text,
                "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(np.random.randint(0, 160), unit="D")
            })
            
    df_ratings = pd.DataFrame(ratings_data)
    df_ratings.to_csv("data/raw_ratings.csv", index=False)
    print(f"Generated {len(df_ratings)} interactions saved to data/raw_ratings.csv")

if __name__ == "__main__":
    main()
