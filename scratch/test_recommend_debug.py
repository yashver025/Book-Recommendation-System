import os
import sys
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.hybrid import HybridRecommendationEngine

def debug():
    print("Debugging hybrid recommendations...")
    engine = HybridRecommendationEngine()
    try:
        engine.load_models()
        print("Models loaded successfully. Generating recommendations...")
        # Get first user ID in df_users
        u_id = int(engine.df_users.iloc[0]["user_id"])
        print(f"Testing user ID: {u_id}")
        recs = engine.recommend(user_id=u_id, top_n=10)
        print("Success! Recommendations generated:")
        for r in recs[:3]:
            print(f"- {r['title']} (Score: {r['score']:.4f})")
    except Exception as e:
        print("Error encountered:")
        traceback.print_exc()

if __name__ == "__main__":
    debug()
