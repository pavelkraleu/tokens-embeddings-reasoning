import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from vertexai.language_models import TextEmbeddingModel


class BaseEmbeddingModel(ABC):
    NAME = "EmbeddingModel"

    def get_cached_embedding(self, text) -> int | None:
        md5_hash = hashlib.md5(text.encode()).hexdigest()

        destination_path = Path("embedding_cache") / f"{md5_hash}_{self.NAME.replace('/','-')}.json"

        if not destination_path.exists():
            embedding = self._compute_embedding(text)
            destination_path.write_text(json.dumps(embedding))

        return json.loads(destination_path.read_text())

    @abstractmethod
    def _compute_embedding(self, text: str) -> list[float]:
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def _compute_embedding(self, text: str) -> list[float]:
        return OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_KEY"], model=self.NAME).embed_query(text)


class GeminiEmbeddingModel(BaseEmbeddingModel):
    def _compute_embedding(self, text: str) -> list[float]:
        m = TextEmbeddingModel.from_pretrained(self.NAME)
        embeddings = m.get_embeddings([text], auto_truncate=False)
        embedding = embeddings[0]

        return embedding.values


class HuggingFaceModel(BaseEmbeddingModel):
    def _compute_embedding(self, text: str) -> list[float]:
        embed_model = HuggingFaceEmbedding(model_name=self.NAME)
        return embed_model.get_text_embedding("query: "+text)


class E5LargeHuggingFaceModel(HuggingFaceModel):
    NAME = "intfloat/multilingual-e5-large"


class E5SmallHuggingFaceModel(HuggingFaceModel):
    NAME = "intfloat/multilingual-e5-small"


class E5SmallEnV2HuggingFaceModel(HuggingFaceModel):
    NAME = "intfloat/e5-small-v2"


class E5SmallEnHuggingFaceModel(HuggingFaceModel):
    NAME = "intfloat/e5-small"


class OpenAIEmbeddingModelAda2(OpenAIEmbeddingModel):
    NAME = "text-embedding-ada-002"


class OpenAIEmbeddingModel3Small(OpenAIEmbeddingModel):
    NAME = "text-embedding-3-small"


class OpenAIEmbeddingModel3Large(OpenAIEmbeddingModel):
    NAME = "text-embedding-3-large"


class GeminiEmbeddingModelGeckoMultilingual(GeminiEmbeddingModel):
    NAME = "textembedding-gecko-multilingual@latest"


class GeminiEmbeddingModelGecko(GeminiEmbeddingModel):
    NAME = "textembedding-gecko@latest"


all_embedding_models = [
    OpenAIEmbeddingModelAda2(),
    OpenAIEmbeddingModel3Small(),
    OpenAIEmbeddingModel3Large(),
    GeminiEmbeddingModelGeckoMultilingual(),
    GeminiEmbeddingModelGecko(),
    E5LargeHuggingFaceModel(),
    E5SmallHuggingFaceModel(),
    E5SmallEnV2HuggingFaceModel(),
    E5SmallEnHuggingFaceModel()
]