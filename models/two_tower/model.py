import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    def __init__(self, num_users, num_genres, num_authors, embedding_dim=32, output_dim=128):
        super(UserTower, self).__init__()
        # Embedding Layers
        self.user_embed = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        self.genre_embed = nn.Embedding(num_genres + 1, embedding_dim, padding_idx=0)
        self.author_embed = nn.Embedding(num_authors + 1, embedding_dim, padding_idx=0)
        
        # Dense Layers
        input_dim = embedding_dim * 3
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, user_ids, fav_genres_multihot, fav_authors_multihot):
        # 1. User ID Embedding
        u_emb = self.user_embed(user_ids) # [batch_size, embedding_dim]
        
        # 2. Favorite Genres Embedding (Average of the genres)
        # Multiply multihot with embedding weights
        # fav_genres_multihot: [batch_size, num_genres]
        # genre_embed weights: [num_genres + 1, embedding_dim]
        # We skip the padding index (0) by offsetting index map by 1 in dataset loader
        g_weights = self.genre_embed.weight[1:] # [num_genres, embedding_dim]
        g_emb = torch.matmul(fav_genres_multihot, g_weights) # [batch_size, embedding_dim]
        # Normalize by count to get average
        g_counts = fav_genres_multihot.sum(dim=1, keepdim=True).clamp(min=1.0)
        g_emb = g_emb / g_counts
        
        # 3. Favorite Authors Embedding (Average)
        a_weights = self.author_embed.weight[1:] # [num_authors, embedding_dim]
        a_emb = torch.matmul(fav_authors_multihot, a_weights)
        a_counts = fav_authors_multihot.sum(dim=1, keepdim=True).clamp(min=1.0)
        a_emb = a_emb / a_counts
        
        # Concatenate embeddings
        x = torch.cat([u_emb, g_emb, a_emb], dim=1)
        
        # Pass through dense layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        
        # L2 Normalize the output to get cosine similarity by dot product
        x = F.normalize(x, p=2, dim=1)
        return x

class BookTower(nn.Module):
    def __init__(self, num_books, num_genres, num_authors, tfidf_dim, embedding_dim=32, output_dim=128):
        super(BookTower, self).__init__()
        self.book_embed = nn.Embedding(num_books + 1, embedding_dim, padding_idx=0)
        self.author_embed = nn.Embedding(num_authors + 1, embedding_dim, padding_idx=0)
        self.genre_embed = nn.Embedding(num_genres + 1, embedding_dim, padding_idx=0)
        
        # Description TF-IDF projection
        self.desc_proj = nn.Linear(tfidf_dim, 128)
        self.desc_bn = nn.BatchNorm1d(128)
        
        # Dense Layers
        input_dim = embedding_dim * 3 + 128
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_dim)
        
    def forward(self, book_ids, author_ids, genres_multihot, desc_tfidf):
        # 1. Book ID Embedding
        b_emb = self.book_embed(book_ids)
        
        # 2. Author Embedding
        a_emb = self.author_embed(author_ids)
        
        # 3. Genre Embedding (Average)
        g_weights = self.genre_embed.weight[1:]
        g_emb = torch.matmul(genres_multihot, g_weights)
        g_counts = genres_multihot.sum(dim=1, keepdim=True).clamp(min=1.0)
        g_emb = g_emb / g_counts
        
        # 4. Description Projection
        d_proj = F.relu(self.desc_bn(self.desc_proj(desc_tfidf)))
        
        # Concatenate features
        x = torch.cat([b_emb, a_emb, g_emb, d_proj], dim=1)
        
        # Pass through dense layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        
        # L2 Normalize
        x = F.normalize(x, p=2, dim=1)
        return x

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_books, num_genres, num_authors, tfidf_dim, embedding_dim=32, output_dim=128):
        super(TwoTowerModel, self).__init__()
        self.user_tower = UserTower(num_users, num_genres, num_authors, embedding_dim, output_dim)
        self.book_tower = BookTower(num_books, num_genres, num_authors, tfidf_dim, embedding_dim, output_dim)
        
    def forward(self, user_inputs, book_inputs):
        # user_inputs: dict of (user_ids, fav_genres, fav_authors)
        # book_inputs: dict of (book_ids, author_ids, genres, desc_tfidf)
        user_emb = self.user_tower(
            user_inputs["user_id"], 
            user_inputs["fav_genres"], 
            user_inputs["fav_authors"]
        )
        book_emb = self.book_tower(
            book_inputs["book_id"], 
            book_inputs["author_id"], 
            book_inputs["genres"], 
            book_inputs["desc_tfidf"]
        )
        return user_emb, book_emb
