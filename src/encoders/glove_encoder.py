import numpy as np
from .text_encoder import TextEncoder
import os

class GloveEncoder(TextEncoder):
    """GloVe embeddings encoder wrapper."""
    
    def __init__(self):
        """Initialize with GloVe path."""

        # Load the GloVe embeddings
        print("Loading GloVe embeddings...")
        path = os.path.expanduser("~/Downloads/glove.6B.300d.txt")
        self._embeddings = self._load_glove(path)
    
    def _load_glove(self, path):
        """Load GloVe embeddings from a file."""
        embedding_dict = {}
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embedding_dict[word] = vector
        return embedding_dict

    def fit(self, texts, y=None):
        """No fitting required."""
        pass

    def transform(self, texts):
        """Transform the texts into GloVe embeddings."""

        vectors = []
        for text in texts:
            tokens = text.split()
            vecs = [self._embeddings[t] for t in tokens if t in self._embeddings]
            if vecs:
                vectors.append(np.mean(vecs, axis=0))
            else:
                vectors.append(np.zeros(len(next(iter(self._embeddings.values())))))
        return np.array(vectors)
