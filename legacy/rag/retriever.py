import sqlite3
from typing import List, Tuple
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import DB_PATH


class TfidfRetriever:
    """
    A simple TF-IDF based retriever.
    """
    def __init__(self, texts: List[str]):
        self.texts = texts
        # Stop words none to allow searching for common phrases if needed,
        # but max_features limits noise.
        self.vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2), max_features=20_000)
        if texts:
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.matrix = None

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        Search for the most similar texts to the query.
        Returns a list of (index, similarity_score, text).
        """
        if self.matrix is None or not self.texts:
            return []

        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).flatten()

        # Get top-k indices
        # We use argpartition for efficiency if k is small compared to N,
        # but argsort is fine for this scale.
        idxs = sims.argsort()[::-1][:top_k]

        results = []
        for i in idxs:
            results.append((int(i), float(sims[i]), self.texts[int(i)]))
        return results


def load_chunks_from_sqlite(db_path: Path = DB_PATH) -> List[str]:
    """Helper to load all text chunks from the SQLite DB."""
    if not db_path.exists():
        print(f"Warning: Database not found at {db_path}")
        return []

    con = sqlite3.connect(db_path.as_posix())
    try:
        cur = con.cursor()
        cur.execute("SELECT text FROM chunks ORDER BY chunk_index ASC")
        rows = cur.fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


class RAGPipeline:
    """
    Combines the retriever and a format helper.
    """
    def __init__(self, db_path: Path = DB_PATH):
        self.chunks = load_chunks_from_sqlite(db_path)
        self.retriever = TfidfRetriever(self.chunks)

    def search(self, query: str, top_k: int = 3):
        return self.retriever.search(query, top_k=top_k)

    @staticmethod
    def format_augmented_prompt(user_prompt: str, retrieved_context: str) -> str:
        """
        Formats the prompt for the model including retrieved context.
        """
        return (
            "Retrieved Context:\n"
            f"{retrieved_context}\n\n"
            "User Prompt:\n"
            f"{user_prompt}\n\n"
            "Answer:\n"
        )
