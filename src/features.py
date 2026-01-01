from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TextFeaturizer:
    vectorizer: TfidfVectorizer

    @staticmethod
    def fit(texts: List[str], max_features: int = 20000, ngram_range: Tuple[int, int] = (1, 2)) -> "TextFeaturizer":
        vec = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=2
        )
        vec.fit(texts)
        return TextFeaturizer(vectorizer=vec)

    def transform(self, texts: List[str]):
        return self.vectorizer.transform(texts)

    def save(self, path: str) -> None:
        joblib.dump(self.vectorizer, path)

    @staticmethod
    def load(path: str) -> "TextFeaturizer":
        vec = joblib.load(path)
        return TextFeaturizer(vectorizer=vec)
