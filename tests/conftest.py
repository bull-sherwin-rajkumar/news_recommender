import re
from types import SimpleNamespace

import pytest

from news_recommender.config import Settings
from news_recommender.ingestion import ingest
from news_recommender.storage import Store, Vectors


class WordTokenizer:
    def __init__(self):
        self.vocab = {}

    def __call__(self, text, text_pair=None, **kwargs):
        matches = list(re.finditer(r"\w+|[^\w\s]", text))
        result = {
            "offset_mapping": [(match.start(), match.end()) for match in matches],
            "input_ids": self.encode(text),
        }
        if text_pair:
            result["input_ids"] += self.encode(text_pair)
        return result

    def encode(self, text, add_special_tokens=True, **kwargs):
        tokens = re.findall(r"\w+|[^\w\s]", text)
        ids = [self.vocab.setdefault(token, len(self.vocab) + 1000) for token in tokens]
        return [101, *ids, 102] if add_special_tokens else ids

    def decode(self, ids, **kwargs):
        inverse = {value: key for key, value in self.vocab.items()}
        return " ".join(inverse[value] for value in ids)


class FakeNLP:
    def __init__(self):
        self.tokenizer = WordTokenizer()
        self.encoder = SimpleNamespace(get_sentence_embedding_dimension=lambda: 2)
        self.ranker = SimpleNamespace(tokenizer=self.tokenizer)

    def document(self, headline, text, tokenizer=None):
        return headline + "\n" + text

    def embed_documents(self, texts):
        return [[1.0, 0.0] if "flood" in text.lower() else [0.0, 1.0] for text in texts]

    def embed_queries(self, texts):
        return self.embed_documents(texts)

    def rerank(self, pairs):
        return [
            4.0
            if any(word in passage.lower() for word in query.lower().split() if len(word) >= 4)
            else -5.0
            for query, passage in pairs
        ]


def source_row(index, *, body=None, section="world", category="news"):
    return {
        "Index": index,
        "Headline": f"Coastal flood report {index}",
        "Article text": body or f"Flood protection and coastal flooding report number {index}.",
        "Date published": "2021-07-15 02:46:59",
        "Url": f"https://example.com/story/{index}",
        "Section": section,
        "Category": category,
        "Author": "Fixture reporter",
        "Keywords": "coast, flood",
        "Description": "Fixture article",
    }


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "index")


@pytest.fixture
def nlp():
    return FakeNLP()


@pytest.fixture
def indexed(settings, nlp):
    vectors = Vectors(settings, create=True)
    ingest(
        settings,
        rows=[source_row(index) for index in range(7)],
        nlp=nlp,
        vectors=vectors,
    )
    return Store(settings.database), vectors, nlp
