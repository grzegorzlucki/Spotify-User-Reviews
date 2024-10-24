import re
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

# nltk.download('punkt')
# nltk.download('stopwords')

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def pre_preparation(self, text: str) -> str:
        text = re.sub(r'Ã[\x80-\xBF]+', ' ', text) 
        text = re.sub(r'[^a-zA-Z\s]', ' ', text) 
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def stopwords_removal(self, words: list) -> list:
        stop_words = stopwords.words('english')
        return [word for word in words if word not in stop_words]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [" ".join(self.stopwords_removal(nltk.word_tokenize(self.pre_preparation(doc)))) for doc in X]