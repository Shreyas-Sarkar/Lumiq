import cohere

from config import settings


class EmbeddingService:
    def __init__(self):
        if not settings.cohere_api_key:
            raise ValueError("COHERE_API_KEY not set")
        self.client = cohere.Client(settings.cohere_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        print("[EMBED] Generating embeddings:", len(texts))
        response = self.client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document",
        )

        return response.embeddings
