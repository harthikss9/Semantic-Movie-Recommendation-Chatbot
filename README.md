# 🎬 Semantic Movie Recommendation Bot

A semantic-aware movie recommendation system that leverages **Sentence-BERT embeddings**, **UMAP + HDBSCAN clustering**, and **CrossEncoder reranking** to suggest movies based on natural language plot descriptions. The bot is deployed on **Telegram** for real-time interaction.

---

## 🔗 Dataset

Used [Millions of Movies](https://www.kaggle.com/datasets/akshaypawar7/millions-of-movies) from Kaggle, containing over **700,000 movies** with metadata fields like `title`, `overview`, `genres`, `keywords`, `tagline`, etc.

---

## 🚀 Features

- **Semantic Search**: Finds semantically similar movies using sentence embeddings from `all-MiniLM-L6-v2`.
- **Clustering**: Uses `UMAP` (dimensionality reduction) and `HDBSCAN` (unsupervised clustering) to accelerate lookup.
- **Re-ranking**: Applies `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank cosine-selected candidates.
- **Telegram Integration**: Responds to user queries with movie recommendations directly inside a Telegram chat.

---

## 📦 Architecture Overview

```text
User Query
   ↓
Sentence-BERT → Cosine Similarity → Top-30 Matches
   ↓
Cross-Encoder Re-ranking → Top-5 Movies
   ↓
Telegram Bot Response
