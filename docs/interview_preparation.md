# Recommendation Systems - Technical Interview Preparation Guide

This document explains the technical design choices, trade-offs, and engineering decisions implemented in the **Novalis** Book Recommendation System. It is formatted to prepare candidates for Machine Learning Engineer (MLE) - Recommendation Systems interviews at companies like Netflix, Spotify, Amazon, YouTube, and Pratilipi.

---

## 1. Core Architectural Pattern: Two-Stage Recommendation System

### Question: Why not score all catalog items directly using a complex model like a deep neural network or XGBoost?
- **Scale and Latency Trade-Off**: Running a complex model with dozens of features (like XGBoost or a deep neural network) over millions of books takes too long, violating the sub-50ms latency requirement of web APIs.
- **The Two-Stage Solution**:
  1. **Stage 1: Candidate Retrieval (Filtering/Retrieval)**: Fast, high-recall, low-latency. Reduces the search space from $10^6$ items to $\sim 10^2$ candidates in $<5$ ms. (Implemented via FAISS vector search, TF-IDF cosine, and SVD).
  2. **Stage 2: Candidate Ranking (Scoring)**: High-precision, complex model. Scores the $\sim 10^2$ candidates using rich features (user-item overlaps, ratings, model predictions) in $<10$ ms. (Implemented via XGBoost Ranker).

---

## 2. Deep Learning: Two-Tower Neural Network Architecture

### Question: What is a Two-Tower model, and why is it preferred for retrieval?
- **Architecture**: It splits feature processing into two separate, parallel sub-networks:
  - **User Tower**: Maps user ID, favorite genres, and history into a 128d embedding vector $\vec{u}$.
  - **Book Tower**: Maps book ID, genres, author, and description TF-IDF into a 128d embedding vector $\vec{b}$.
- **Inference Efficiency**:
  - Because the towers are independent, book embeddings $\vec{b}$ can be precomputed offline and loaded into a vector search index (like FAISS).
  - At query time, the system only needs to pass the active user features through the **User Tower** to get $\vec{u}$ (takes $<1$ ms), then query the FAISS index to find books maximizing the inner product:
    $$\text{Score} = \vec{u} \cdot \vec{b}$$
- **Loss Design**: We train with **Binary Cross-Entropy (BCE) with Negative Sampling**. Positive interactions (views/likes) get label 1; randomly sampled uninteracted books get label 0. A temperature scale ($\tau = 10$) is applied to the cosine similarity scores to prevent gradient saturation during backpropagation.

---

## 3. Vector Search: Facebook AI Similarity Search (FAISS)

### Question: How does FAISS scale vector retrieval, and how is cosine similarity calculated?
- **Cosine Similarity via Dot Product**: Cosine similarity is defined as:
  $$\text{CosineSimilarity}(\vec{u}, \vec{b}) = \frac{\vec{u} \cdot \vec{b}}{\|\vec{u}\|_2 \|\vec{b}\|_2}$$
  By L2-normalizing the output embeddings of both towers in the PyTorch model (forcing $\|\vec{u}\|_2 = 1$ and $\|\vec{b}\|_2 = 1$), the cosine similarity simplifies to a simple dot product ($\vec{u} \cdot \vec{b}$). This allows us to use `faiss.IndexFlatIP` (Inner Product index) which is extremely fast and natively calculated.
- **Scalability**: While `IndexFlatIP` does brute-force comparison (flat index), in production we can swap it for:
  - `IndexIVFFlat` (Inverted File Index): clusters vectors and restricts search to the closest cluster centroids (reduces query complexity from $O(N)$ to $O(\log N)$).
  - `IndexHNSW` (Hierarchical Navigable Small World): builds a graph representation for approximate nearest neighbors.

---

## 4. Collaborative Filtering: SVD vs. Neighborhood Methods (KNN)

### Question: Why prefer Matrix Factorization (SVD) over neighborhood collaborative filtering (KNN)?
- **Dimensionality Reduction**: SVD projects sparse interaction matrices into low-dimensional dense matrices (capturing latent factors), resolving the sparsity problem.
- **Complexity**:
  - Neighborhood methods (KNN) require computing and storing an $O(U^2)$ or $O(I^2)$ similarity matrix, which becomes computationally prohibitive as the user base grows.
  - SVD inference is a simple dot product of precomputed latent vectors, making it highly scalable:
    $$\hat{r}_{u,i} = \mu + b_u + b_i + p_u \cdot q_i$$

---

## 5. Ranking Layer: Gradient Boosted Trees (XGBoost)

### Question: Why use XGBoost as a ranker rather than logistic regression or a deep neural network?
- **Non-Linear Relationships**: XGBoost handles highly non-linear feature interactions (e.g. "if user favorite genre matches book genre AND average rating > 4.2, raise rank").
- **Robust Feature Handling**: It natively handles features on different scales (similarity scores $[0,1]$, average ratings $[1,5]$, ratings counts $[0, 10^5]$) without scaling/normalization, and handles missing values gracefully.
- **Feature Ensembling**: It allows us to feed raw predictions from other models (SVD prediction, Two-tower similarity, TF-IDF similarity) as features. The ranker acts as a meta-classifier, learning how to combine these signals optimal.

---

## 6. Cold Start Strategies

### Question: How does the system handle Cold Start for new users and new items?
- **User Cold Start**: New users have no historical rating logs. We present an **interactive preference questionnaire** (selecting favorite genres, authors). We map these selections to a metadata-based similarity score, combining genre match, author match, and global popularity to rank the catalog.
- **Item Cold Start**: New books have no user ratings, making them invisible to collaborative filtering (SVD). We fall back on **Content-Based metadata representations** (TF-IDF plots + Book Tower embedding from the Two-Tower model) to position the book in the vector space relative to user interests.

---

## 7. Explainable AI (XAI) in Recommenders

### Question: How are recommendation explanations generated?
- **Multi-Source Attribution**: We use a rule-based post-processing explainer that parses the feature importances of the XGBoost ranker. If the Jaccard genre match is high, we state genre overlap; if the author matches, we highlight author affinity; if the SVD score dominates, we attribute it to collaborative consensus.

---

## 8. Offline Evaluation Metrics Selection

### Question: Which metrics do you track for retrieval, ranking, and predictions?
- **Prediction Accuracy**: **RMSE** and **MAE** to track SVD rating prediction errors.
- **Retrieval Coverage**: **Recall@K**, **Precision@K**, and **Hit Rate@K** to measure if Stage 1 successfully retrieves items the user eventually interacted with.
- **Ranking Quality**:
  - **NDCG@K (Normalized Discounted Cumulative Gain)**: penalizes placing relevant items lower down the list.
  - **MAP@K (Mean Average Precision)**: evaluates precision at each rank position.
  - **MRR (Mean Reciprocal Rank)**: focuses on how quickly the first relevant recommendation appears.
