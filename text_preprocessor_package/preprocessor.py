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
        text = text.lower()
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
    
    def main(self, text_list):
        processed_texts = []
        for text in text_list:
            cleaned_text = self.pre_preparation(text)
            tokens = nltk.word_tokenize(cleaned_text)
            filtered_tokens = self.stopwords_removal(tokens)
            processed_text = " ".join(filtered_tokens)
            processed_texts.append(processed_text)
        return processed_texts