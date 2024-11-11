import re
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def pre_preparation(self, text: str) -> str:
        text = re.sub(r'Ã[\x80-\xBF]+', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
        return text

    def stopwords_removal(self, words: list) -> list:
        return [word for word in words if word not in self.stop_words]

    def process_text(self, text: str) -> str:
        cleaned_text = self.pre_preparation(text)
        tokens = nltk.word_tokenize(cleaned_text)
        filtered_tokens = self.stopwords_removal(tokens)
        return " ".join(filtered_tokens)

    def main(self, text_list: list) -> list:
        return [self.process_text(text) for text in text_list]
