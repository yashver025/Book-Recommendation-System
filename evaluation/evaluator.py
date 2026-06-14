import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def compute_ndcg_at_k(recommended_ids, positive_ids, k):
    """
    Computes NDCG@K.
    """
    recommended_ids = recommended_ids[:k]
    dcg = 0.0
    for idx, item in enumerate(recommended_ids):
        if item in positive_ids:
            dcg += 1.0 / np.log2(idx + 2)
            
    idcg = 0.0
    ideal_length = min(len(positive_ids), k)
    for idx in range(ideal_length):
        idcg += 1.0 / np.log2(idx + 2)
        
    return dcg / idcg if idcg > 0.0 else 0.0

def compute_map_at_k(recommended_ids, positive_ids, k):
    """
    Computes MAP@K.
    """
    recommended_ids = recommended_ids[:k]
    avg_precision = 0.0
    num_hits = 0.0
    
    for idx, item in enumerate(recommended_ids):
        if item in positive_ids:
            num_hits += 1.0
            avg_precision += num_hits / (idx + 1)
            
    return avg_precision / min(len(positive_ids), k) if positive_ids else 0.0

def compute_mrr(recommended_ids, positive_ids):
    """
    Computes MRR (Mean Reciprocal Rank).
    """
    for idx, item in enumerate(recommended_ids):
        if item in positive_ids:
            return 1.0 / (idx + 1)
    return 0.0

def evaluate_system(engine, test_df, k=10):
    print(f"Evaluating Hybrid Recommendation System (K={k}) on test set...")
    
    # 1. Prediction Metrics (Surprise SVD on test interactions with ratings)
    test_ratings = test_df.dropna(subset=["rating"])
    svd_errors = []
    abs_errors = []
    
    for _, row in test_ratings.iterrows():
        u_id = row["user_id"]
        b_id = row["book_id"]
        actual = row["rating"]
        
        try:
            pred = engine.collab_rec.predict_rating(u_id, b_id)
            svd_errors.append((actual - pred) ** 2)
            abs_errors.append(abs(actual - pred))
        except Exception:
            pass
            
    rmse = np.sqrt(np.mean(svd_errors)) if svd_errors else np.nan
    mae = np.mean(abs_errors) if abs_errors else np.nan
    
    # 2. Ranking and Retrieval Metrics
    # Positive items in test set: rating >= 4 or positive implicit interaction
    user_pos_test = test_df[
        (test_df["rating"] >= 4) | (test_df["interaction_type"].isin(["like", "shelve"]))
    ].groupby("user_id")["book_id"].apply(set).to_dict()
    
    df_train = pd.read_csv("data/train_interactions.csv")
    
    recalls = []
    precisions = []
    hit_rates = []
    ndcgs = []
    maps = []
    mrrs = []
    
    test_users = list(user_pos_test.keys())
    # Evaluate on a subset of 100 users if test set is too large to save run time
    eval_users = test_users[:100]
    
    print(f"Evaluating ranking metrics for {len(eval_users)} test users...")
    
    for u_id in eval_users:
        pos_items = user_pos_test[u_id]
        if not pos_items:
            continue
            
        try:
            # Get user training rated books
            user_train = df_train[df_train["user_id"] == u_id]
            train_rated_ids = user_train["book_id"].unique().tolist()
            
            # Generate recommendations using hybrid engine (filtering only train ratings)
            recs = engine.recommend(user_id=u_id, top_n=k, filter_rated_book_ids=train_rated_ids)
            rec_ids = [r["book_id"] for r in recs]
            
            # Print debug info for first 5 users
            if len(recalls) < 5:
                print(f"User {u_id}: generated {len(recs)} recommendations. Target positive items: {list(pos_items)[:5]}")
                if recs:
                    print(f"  First recommendation: {recs[0]['title']} (ID: {recs[0]['book_id']})")
            
            # Hits
            hits = [item for item in rec_ids if item in pos_items]
            
            # Recall@K = Hits / Total Positive Items
            recall = len(hits) / len(pos_items)
            recalls.append(recall)
            
            # Precision@K = Hits / K
            precision = len(hits) / k
            precisions.append(precision)
            
            # HitRate@K = 1 if Hits > 0 else 0
            hit_rates.append(1.0 if hits else 0.0)
            
            # NDCG@K
            ndcgs.append(compute_ndcg_at_k(rec_ids, pos_items, k))
            
            # MAP@K
            maps.append(compute_map_at_k(rec_ids, pos_items, k))
            
            # MRR
            mrrs.append(compute_mrr(rec_ids, pos_items))
            
        except Exception as e:
            import traceback
            print(f"Error evaluating user {u_id}: {e}")
            traceback.print_exc()
            
    avg_recall = np.mean(recalls) if recalls else 0.0
    avg_precision = np.mean(precisions) if precisions else 0.0
    avg_hit_rate = np.mean(hit_rates) if hit_rates else 0.0
    avg_ndcg = np.mean(ndcgs) if ndcgs else 0.0
    avg_map = np.mean(maps) if maps else 0.0
    avg_mrr = np.mean(mrrs) if mrrs else 0.0
    
    report_content = f"""# Hybrid Recommendation System Evaluation Report

This report summarizes the offline performance of the Two-Stage Hybrid Recommendation System.

## Prediction Metrics (Collaborative Filtering)
- **Root Mean Squared Error (RMSE)**: {rmse:.4f}
- **Mean Absolute Error (MAE)**: {mae:.4f}
*Note: Evaluates accuracy of explicit ratings predictions (SVD).*

## Retrieval Metrics (Stage 1)
- **Precision@{k}**: {avg_precision:.4f}
- **Recall@{k}**: {avg_recall:.4f}
- **Hit Rate@{k}**: {avg_hit_rate:.4f}
*Note: Evaluates candidate generation coverage on user positive test items.*

## Ranking Metrics (Stage 2)
- **NDCG@{k} (Normalized Discounted Cumulative Gain)**: {avg_ndcg:.4f}
- **MAP@{k} (Mean Average Precision)**: {avg_map:.4f}
- **MRR (Mean Reciprocal Rank)**: {avg_mrr:.4f}
*Note: Evaluates accuracy of candidate ordering based on user relevance.*
"""
    os.makedirs("reports", exist_ok=True)
    with open("reports/system_evaluation_report.md", "w") as f:
        f.write(report_content)
        
    print("System evaluation complete. Report written to reports/system_evaluation_report.md")
    print(report_content)
    
if __name__ == "__main__":
    from models.hybrid import HybridRecommendationEngine
    if os.path.exists("data/test_interactions.csv"):
        df_test = pd.read_csv("data/test_interactions.csv")
        engine = HybridRecommendationEngine()
        try:
            engine.load_models()
            evaluate_system(engine, df_test, k=10)
        except Exception as e:
            print(f"Skipping main runner evaluation: models not fully trained yet. ({e})")
