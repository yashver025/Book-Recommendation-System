import os
import sys
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path for local engine fallback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try loading the hybrid engine locally as fallback
engine = None
try:
    from models.hybrid import HybridRecommendationEngine
    from database.db import SessionLocal
    from database.models import User, Book, Rating, UserPreference
except Exception as e:
    print(f"Local import error (not fatal if API is used): {e}")

# API Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Set page config
st.set_page_config(
    page_title="Novalis - AI Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0B0E17 0%, #151A2C 100%);
        color: #E2E8F0;
    }
    
    /* Premium Headers */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #A855F7 0%, #6366F1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Glassmorphic Cards */
    .book-card {
        background: rgba(30, 41, 59, 0.45);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }
    
    .book-card:hover {
        transform: translateY(-5px);
        border-color: rgba(168, 85, 247, 0.4);
        box-shadow: 0 10px 30px rgba(168, 85, 247, 0.15);
    }
    
    /* Genre tags */
    .genre-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #A855F7;
        margin-bottom: 4px;
    }
    
    /* Rating display */
    .rating-stars {
        color: #FBBF24;
        font-weight: 600;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to query backend or fallback
@st.cache_resource
def get_hybrid_engine():
    global engine
    if engine is None:
        try:
            engine = HybridRecommendationEngine()
            engine.load_models()
        except Exception as e:
            st.warning(f"Could not load local recommendation engine: {e}")
    return engine

def check_backend():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json()
    except Exception:
        pass
    return False, None

def get_recommendations_api(user_id, top_n=10):
    try:
        response = requests.get(f"{API_URL}/recommend/{user_id}?top_n={top_n}", timeout=5)
        if response.status_code == 200:
            return response.json()["recommendations"]
        else:
            st.error(f"Error from API: {response.json().get('detail', 'Unknown error')}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
    return []

# Sidebar Navigation
st.sidebar.markdown("<h1 style='text-align: center;'>NOVALIS AI</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #94A3B8; font-size: 13px;'>Next-Gen Hybrid Book Recommender</p>", unsafe_allow_html=True)
st.sidebar.write("---")

page = st.sidebar.selectbox(
    "Navigate",
    ["Home Catalog", "User Dashboard", "Personalized Recommendations", "Model Analytics"]
)

# API vs Local mode check
backend_alive, health_data = check_backend()

# Fixed Optimal Hybrid Engine Weights
weights = {
    "content": 0.20,
    "collaborative": 0.30,
    "two_tower": 0.20,
    "ranking": 0.30
}

# Initialize Database session for local queries
@st.cache_resource
def get_local_db():
    try:
        return SessionLocal()
    except Exception:
        return None

db_session = get_local_db()

# Ensure we have access to metadata
df_books = None
if os.path.exists("data/processed_books.csv"):
    df_books = pd.read_csv("data/processed_books.csv")
    
# -------------------------------------------------------------
# PAGE 1: HOME CATALOG
# -------------------------------------------------------------
if page == "Home Catalog":
    st.title("📚 Book Discovery Catalog")
    st.write("Browse, search, and examine titles across our deep library.")
    
    if df_books is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            default_search = st.session_state.get("search_query", "")
            search_query = st.text_input("🔍 Search books by title, author, or description...", value=default_search)
            st.session_state.search_query = search_query
        with col2:
            all_genres = sorted(list(set([g for g_str in df_books["standardized_genres"].dropna() for g in g_str.split("|")])))
            selected_genre = st.selectbox("📂 Filter by Genre", ["All"] + [g.capitalize() for g in all_genres])
            
        # Filter logic
        filtered_df = df_books.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df["title"].str.contains(search_query, case=False, na=False) |
                filtered_df["author"].str.contains(search_query, case=False, na=False) |
                filtered_df["description"].str.contains(search_query, case=False, na=False)
            ]
        if selected_genre != "All":
            filtered_df = filtered_df[filtered_df["standardized_genres"].str.contains(selected_genre.lower(), na=False)]
            
        if len(filtered_df) == 0:
            st.warning("⚠️ No matching books found. Real-world titles (like *The Alchemist*) are not in this synthetic dataset.")
            st.info("👇 Click any of these popular synthetic titles to instantly search for them:")
            
            # Select 5 consistent or popular titles from the dataset to guide the user
            sample_titles = ["The Shadow of Dragon", "12 Rules of Goals", "The Era of War", "Hearts in Passion", "Mastering your Habits"]
            sample_books = df_books[df_books["title"].isin(sample_titles)].head(5)
            if len(sample_books) < 5:
                sample_books = df_books.head(5)
                
            cols_btn = st.columns(len(sample_books))
            for idx_btn, (_, sample_row) in enumerate(sample_books.iterrows()):
                if cols_btn[idx_btn].button(sample_row['title'], key=f"sample_{sample_row['book_id']}"):
                    st.session_state.search_query = sample_row['title']
                    st.rerun()
        else:
            st.write(f"Showing {len(filtered_df)} matching books")
        
        # Paginate results (show 10 at a time)
        books_per_page = 6
        num_pages = max(1, int(np.ceil(len(filtered_df) / books_per_page)))
        page_num = st.number_input("Page", min_value=1, max_value=num_pages, value=1, step=1)
        
        start_idx = (page_num - 1) * books_per_page
        end_idx = start_idx + books_per_page
        
        display_df = filtered_df.iloc[start_idx:end_idx]
        
        for idx, row in display_df.iterrows():
            st.markdown(f"""
            <div class='book-card'>
                <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                    <div>
                        <h3 style='margin: 0 0 6px 0; color: #F8FAFC;'>{row['title']}</h3>
                        <p style='color: #A7F3D0; font-weight: 600; margin: 0 0 10px 0;'>by {row['author']}</p>
                    </div>
                    <div class='rating-stars'>⭐ {row['average_rating']:.2f} <span style='color: #64748B; font-size:12px;'>({row['ratings_count']:,} ratings)</span></div>
                </div>
                <div style='margin-bottom: 12px;'>
                    {''.join([f"<span class='genre-tag'>{g}</span>" for g in row['genres'].split('|')])}
                </div>
                <p style='color: #94A3B8; font-size: 14px; line-height: 1.5;'>{row['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive action: show similar books
            if st.button(f"Find Books Similar to '{row['title']}'", key=f"sim_{row['book_id']}"):
                st.subheader(f"Titles Similar to '{row['title']}'")
                
                # Fetch similar books
                sims = []
                if backend_alive:
                    try:
                        res = requests.get(f"{API_URL}/similar-books/{row['book_id']}?top_n=3")
                        if res.status_code == 200:
                            sims = res.json()
                    except Exception:
                        pass
                else:
                    # fallback
                    local_engine = get_hybrid_engine()
                    if local_engine:
                        sims = local_engine.content_rec.recommend_similar_books(book_id=row['book_id'], top_n=3)
                        
                if sims:
                    scol1, scol2, scol3 = st.columns(3)
                    cols = [scol1, scol2, scol3]
                    for idx_s, s in enumerate(sims[:3]):
                        with cols[idx_s]:
                            st.markdown(f"""
                            <div class='book-card' style='padding: 16px; min-height: 180px;'>
                                <h4 style='margin:0 0 4px 0; font-size:15px;'>{s['title']}</h4>
                                <p style='color: #818CF8; font-size:12px; margin:0 0 8px 0;'>{s['author']}</p>
                                <p style='color: #A855F7; font-size:11px; font-weight:600;'>{s['reason']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No similarity profiles loaded.")
    else:
        st.info("Catalog data missing. Please generate the data first.")

# -------------------------------------------------------------
# PAGE 2: USER DASHBOARD
# -------------------------------------------------------------
elif page == "User Dashboard":
    st.title("👤 User Profile Dashboard")
    
    if db_session:
        # Load user options
        users = db_session.query(User).limit(50).all()
        user_opts = {f"{u.username} (ID: {u.id})": u.id for u in users}
        
        # Option to create a new user (Cold Start handling)
        with st.expander("➕ Register New Profile (Cold Start Demonstration)"):
            new_username = st.text_input("Username")
            # Select favorite genres
            genre_choices = ["Fantasy", "Science Fiction", "Mystery", "Thriller", "Romance", "Historical Fiction", "Non-fiction", "Biography", "Self-Help", "Business"]
            fav_genres = st.multiselect("Favorite Genres", genre_choices)
            # Input favorite authors
            author_choices = [
                "J.R.R. Tolkien", "George R.R. Martin", "Isaac Asimov", "Arthur C. Clarke",
                "Agatha Christie", "Stephen King", "Jane Austen", "Malcolm Gladwell", "James Clear"
            ]
            fav_authors = st.multiselect("Favorite Authors", author_choices)
            
            if st.button("Register & Submit Preferences"):
                if new_username:
                    # POST to backend if alive
                    success = False
                    if backend_alive:
                        try:
                            res = requests.post(f"{API_URL}/new-user", json={
                                "username": new_username,
                                "favorite_genres": fav_genres,
                                "favorite_authors": fav_authors
                            })
                            if res.status_code == 201:
                                st.success(f"Registered profile '{new_username}' with ID {res.json()['user_id']}!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"API Register failed: {e}")
                    
                    if not success:
                        # Fallback local insertion
                        try:
                            max_u = db_session.query(User).order_by(User.id.desc()).first()
                            new_id = (max_u.id + 1) if max_u else 1
                            u = User(id=new_id, username=new_username)
                            db_session.add(u)
                            db_session.flush()
                            
                            pref = UserPreference(
                                user_id=new_id,
                                favorite_genres="|".join([g.lower() for g in fav_genres]),
                                favorite_authors="|".join([a.lower() for a in fav_authors])
                            )
                            db_session.add(pref)
                            db_session.commit()
                            st.success(f"Registered profile '{new_username}' locally with ID {new_id}!")
                            st.rerun()
                        except Exception as ex:
                            db_session.rollback()
                            st.error(f"Local Register failed: {ex}")
                else:
                    st.warning("Please fill in username.")
                    
        selected_user_label = st.selectbox("Select User Profile to Load", list(user_opts.keys()))
        user_id = user_opts[selected_user_label]
        
        # Load profile statistics
        user = db_session.query(User).filter(User.id == user_id).first()
        st.subheader(f"Reading & Interaction Log: {user.username}")
        
        ratings = db_session.query(Rating).filter(Rating.user_id == user_id).all()
        pref = db_session.query(UserPreference).filter(UserPreference.user_id == user_id).first()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='book-card'>
                <h4>Profile Summary</h4>
                <p><b>User ID:</b> {user.id}</p>
                <p><b>Username:</b> {user.username}</p>
                <p><b>Preferred Genres:</b> {pref.favorite_genres.replace('|', ', ').title() if pref and pref.favorite_genres else 'None'}</p>
                <p><b>Preferred Authors:</b> {pref.favorite_authors.replace('|', ', ').title() if pref and pref.favorite_authors else 'None'}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class='book-card' style='text-align: center;'>
                <h4>Interaction History</h4>
                <div class='metric-value'>{len(ratings)}</div>
                <p style='color:#94A3B8;'>Total Ratings & Views</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("---")
        st.subheader("📚 Interactive History Log")
        if ratings:
            history_data = []
            for r in ratings:
                book = db_session.query(Book).filter(Book.book_id == r.book_id).first()
                if book:
                    history_data.append({
                        "Title": book.title,
                        "Author": book.author,
                        "Type": r.interaction_type.upper(),
                        "Rating (Stars)": f"{r.rating:.1f}" if r.rating is not None else "N/A",
                        "Review": r.review_text if r.review_text else "No review"
                    })
            st.table(pd.DataFrame(history_data))
        else:
            st.info("This user has no ratings or interaction logs. Running cold start recommendations.")
    else:
        st.info("Database connection not active. Run training pipeline first to build SQLite database.")

# -------------------------------------------------------------
# PAGE 3: RECOMMENDATIONS PAGE
# -------------------------------------------------------------
elif page == "Personalized Recommendations":
    st.title("🎯 Personalized Recommendations Engine")
    st.write("Configured weights are applied directly across SVD, Content, Two-Tower, and XGBoost components.")
    
    if db_session:
        # Load user list
        users = db_session.query(User).limit(50).all()
        user_opts = {f"{u.username} (ID: {u.id})": u.id for u in users}
        selected_user_label = st.selectbox("Select User Profile", list(user_opts.keys()), key="rec_user_sel")
        user_id = user_opts[selected_user_label]
        
        # Recommendations generation trigger
        if st.button("Generate Recommendations", key="gen_rec_btn"):
            st.subheader(f"Top 10 Personalized Book Matches")
            
            recs = []
            # Query backend if alive, passing custom weights isn't standard in GET endpoints,
            # so we run recommendations locally using our custom weight configuration!
            # This directly shows hybrid system capability.
            local_engine = get_hybrid_engine()
            if local_engine:
                with st.spinner("Executing two-stage pipeline..."):
                    recs = local_engine.recommend(user_id=user_id, top_n=10, weights=weights)
                    
            if recs:
                for idx, r in enumerate(recs):
                    st.markdown(f"""
                    <div class='book-card'>
                        <div style='display: flex; justify-content: space-between; align-items: flex-start;'>
                            <div>
                                <h3 style='margin: 0 0 6px 0; color: #F8FAFC;'>{idx+1}. {r['title']}</h3>
                                <p style='color: #A7F3D0; font-weight: 600; margin: 0 0 10px 0;'>by {r['author']}</p>
                            </div>
                            <div class='rating-stars' style='font-size:18px;'>🚀 Match: {r['score']:.2f}</div>
                        </div>
                        <div style='margin-bottom: 12px;'>
                            {''.join([f"<span class='genre-tag'>{g}</span>" for g in r['genres'].split('|')])}
                        </div>
                        <p style='color: #94A3B8; font-size: 14px; line-height: 1.5;'><b>Why Recommended:</b> {r['reason']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Failed to generate recommendations. Ensure models are trained.")
    else:
        st.info("Database connection missing. Complete train_pipeline.py first.")

# -------------------------------------------------------------
# PAGE 4: MODEL ANALYTICS PAGE
# -------------------------------------------------------------
elif page == "Model Analytics":
    st.title("📊 Model Analytics & System Evaluation")
    st.write("Track hyperparameter states, metrics, and global corpus trends.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Collaborative Filtering Metrics")
        if os.path.exists("reports/collaborative_report.txt"):
            with open("reports/collaborative_report.txt", "r") as f:
                st.text(f.read())
        else:
            st.info("SVD/KNN report missing. Run training pipeline.")
            
    with col2:
        st.subheader("🏆 Candidate Retrieval Benchmark")
        if os.path.exists("reports/faiss_benchmark.txt"):
            with open("reports/faiss_benchmark.txt", "r") as f:
                st.text(f.read())
        else:
            st.info("FAISS benchmark report missing. Run training pipeline.")
            
    st.write("---")
    st.subheader("📈 Offline System Validation Reports")
    if os.path.exists("reports/system_evaluation_report.md"):
        with open("reports/system_evaluation_report.md", "r") as f:
            st.markdown(f.read())
    else:
        st.info("System-wide offline validation metrics (Hit Rate, Recall, NDCG) not compiled. Generate them by running the evaluator script.")
        
    st.write("---")
    st.subheader("📚 Book Catalog Analytics")
    if df_books is not None:
        # Genre counts
        genres_list = []
        for g_str in df_books["standardized_genres"].dropna():
            genres_list.extend(g_str.split("|"))
        df_genres = pd.Series(genres_list).value_counts().reset_index()
        df_genres.columns = ["Genre", "Count"]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x="Count", y="Genre", data=df_genres.head(10), palette="rocket", ax=ax)
        ax.set_title("Top 10 Genres in Book Catalog")
        ax.set_xlabel("Number of Books")
        ax.set_ylabel("Genre")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
