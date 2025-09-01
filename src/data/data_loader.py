import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    """Loads and splits data for training."""
    
    def __init__(self, file_path: str):
        """Initialize with data file path."""
        self._file_path = file_path
        self._data = pd.read_csv(self._file_path)
        self._X_train_text = None
        self._X_test_text = None
        self._y_train = None
        self._y_test = None
        
    def prepare(self, test_size=0.2):
        """Split data into train and test sets."""
        self._X_train_text, self._X_test_text, self._y_train, self._y_test = train_test_split(
            self._data['text'],
            self._data['generated'],
            test_size=test_size,
            stratify=self._data['generated']
        )

    def get_train_data(self):
        """Get training data."""
        return self._X_train_text, self._y_train

    def get_test_data(self):
        """Get test data."""
        return self._X_test_text, self._y_test