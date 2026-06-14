# Hybrid Recommendation System Evaluation Report

This report summarizes the offline performance of the Two-Stage Hybrid Recommendation System.

## Prediction Metrics (Collaborative Filtering)
- **Root Mean Squared Error (RMSE)**: 0.8343
- **Mean Absolute Error (MAE)**: 0.6457
*Note: Evaluates accuracy of explicit ratings predictions (SVD).*

## Retrieval Metrics (Stage 1)
- **Precision@10**: 0.0570
- **Recall@10**: 0.2635
- **Hit Rate@10**: 0.5100
*Note: Evaluates candidate generation coverage on user positive test items.*

## Ranking Metrics (Stage 2)
- **NDCG@10 (Normalized Discounted Cumulative Gain)**: 0.2850
- **MAP@10 (Mean Average Precision)**: 0.2260
- **MRR (Mean Reciprocal Rank)**: 0.4483
*Note: Evaluates accuracy of candidate ordering based on user relevance.*
