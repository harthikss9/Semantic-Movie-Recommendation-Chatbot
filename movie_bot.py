from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import logging
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# === Bot Token ===
BOT_TOKEN = "your bot token"

# === Logging ===
logging.basicConfig(level=logging.INFO, filename="bot.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# === Greeting Tracker ===
seen_users = set()

# === Load Data ===
print("📦 Loading data...")
metadata = joblib.load("movie_metadata_umap_clusters.pkl")
embeddings = np.load("movie_weighted_embeddings.npy")

# === Load Models (CPU version) ===
print("🧠 Loading models...")
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")  # Runs on CPU
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# === Recommendation Function ===
def recommend_movies_clustered(query, top_n_cosine=20, top_k_final=5):
    query_embedding = bi_encoder.encode([query])
    sims = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_n_cosine]

    movie_pairs = []
    for idx in top_indices:
        text = f"{metadata[idx]['title']} {metadata[idx]['overview']} {metadata[idx]['keywords']} {metadata[idx]['tagline']}"
        movie_pairs.append((query, text))

    scores = cross_encoder.predict(movie_pairs)
    reranked = sorted(zip(scores, top_indices), reverse=True)[:top_k_final]

    result = "*🎬 Top Movie Recommendations:*\n\n"
    for i, (score, idx) in enumerate(reranked, 1):
        movie = metadata[idx]
        overview = movie['overview'][:300] + "..." if movie['overview'] else "No overview available."
        result += f"*{i}. {movie['title']}*\n_{overview}_\n\n"
    return result

# === Telegram Handler ===
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    query = update.message.text.strip()

    if user_id not in seen_users:
        seen_users.add(user_id)
        await update.message.reply_text("👋 Hi! Send me any movie plot or theme and I’ll find great similar movies!")

    await update.message.reply_text("🔎 Recommending similar movies...")

    try:
        reply = recommend_movies_clustered(query)
        await update.message.reply_text(reply, parse_mode="Markdown")
    except Exception as e:
        logging.exception("❌ Error:")
        await update.message.reply_text(f"❌ Something went wrong: {str(e)}")

# === Run the Bot ===
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🚀 Movie bot running...")
    app.run_polling()

if __name__ == "__main__":
    main()
