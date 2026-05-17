"""
TF-IDF retriever and RAG pipeline for the Dáil LLM project.
"""
import sqlite3
from typing import List, Tuple
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import DB_PATH


class TfidfRetriever:
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2), max_features=20_000)
        if texts:
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.matrix = None

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        if self.matrix is None or not self.texts:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).flatten()
        idxs = sims.argsort()[::-1][:top_k]
        return [(int(i), float(sims[i]), self.texts[int(i)]) for i in idxs]


def load_chunks_from_sqlite(db_path: Path = DB_PATH) -> List[str]:
    if not db_path.exists():
        print(f"Warning: Database not found at {db_path}")
        return []
    con = sqlite3.connect(db_path.as_posix())
    try:
        cur = con.cursor()
        cur.execute("SELECT text FROM chunks ORDER BY chunk_index ASC")
        return [r[0] for r in cur.fetchall()]
    finally:
        con.close()


class RAGPipeline:
    def __init__(self, db_path: Path = DB_PATH):
        self.chunks = load_chunks_from_sqlite(db_path)
        self.retriever = TfidfRetriever(self.chunks)

    def search(self, query: str, top_k: int = 3):
        return self.retriever.search(query, top_k=top_k)

    @staticmethod
    def format_augmented_prompt(user_prompt: str, retrieved_context: str) -> str:
        return (
            "Retrieved Context:\n"
            f"{retrieved_context}\n\n"
            "User Prompt:\n"
            f"{user_prompt}\n\n"
            "Answer:\n"
        )
