from sklearn.feature_extraction.text import TfidfVectorizer
from .text_encoder import TextEncoder

class TfidfEncoder(TextEncoder):
    """Wrapper for the TF-IDF vectorizer."""

    def __init__(self):
        """Initialize with TF-IDF vectorizer."""

        self._vectorizer = TfidfVectorizer()

    def fit(self, texts, y=None):
        """Fit the TF-IDF vectorizer."""
        self._vectorizer.fit(texts)

    def transform(self, texts):
        """Transform the texts into TF-IDF embeddings."""
        return self._vectorizer.transform(texts)
