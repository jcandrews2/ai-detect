import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            max_features=50000,  # Limit vocabulary size
            min_df=2,            # Ignore terms that appear in less than 2 documents
            max_df=0.95          # Ignore terms that appear in more than 95% of documents
        )
        
    def load(self):
        """Load data from CSV file."""
        self.data = pd.read_csv(self.file_path)
        
    def split(self, test_size=0.2):
        """Split data into train and test sets."""
        if self.data is None:
            self.load()
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.data['text'],
            self.data['generated'],
            test_size=test_size,
            stratify=self.data['generated']
        )

        return X_train, X_test, y_train, y_test
    
    def vectorize(self, train_texts, test_texts):
        """Convert text data into TF-IDF vectors."""
        X_train = self.vectorizer.fit_transform(train_texts)
        X_test = self.vectorizer.transform(test_texts)
        return X_train, X_test

    def prepare(self, test_size=0.2):
        """Prepare data for training."""
        X_train_text, X_test_text, self.y_train, self.y_test = self.split(test_size)
        self.X_train, self.X_test = self.vectorize(X_train_text, X_test_text)
        return self

    def get_train_data(self):
        """Get training data."""
        if self.X_train is None:
            raise ValueError("Data must be prepared before access")
        return self.X_train, self.y_train

    def get_test_data(self):
        """Get test data."""
        if self.X_test is None:
            raise ValueError("Data must be prepared before access")
        return self.X_test, self.y_test

    def vectorize_text(self, text):
        """Vectorize a single text or list of texts."""
        if isinstance(text, str):
            text = [text]
        return self.vectorizer.transform(text)