

import faiss
from sentence_transformers import SentenceTransformer

class ReRanker:
    def __init__(self, index):
        self.index = index
        self.model_registry = []


    
    def initialize(model):
        embeddings = model.encode(["Place Holder"], convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        return index

    def generate_embeddings(model, text_list):
        embeddings = model.encode(text_list, convert_to_numpy=True)
        return embeddings

    def store_documents(index, embeddings):
        index.add(embeddings)  # Add all embeddings to index
        return True

    def queryk(index, embedding, k):
        distances, indices = index.search(embedding, k)
        return (distances, indices)


    def rerank(documents, query, k=10):
        """
        Rerank the documents based on the query using the index.
        :param documents: List of documents to rerank.
        :param query: The query to use for reranking.
        :param k: The number of top documents to return.
        :return: List of reranked documents.
        """

        model = SentenceTransformer('all-MiniLM-L6-v2')
        faiss_index = ReRanker.initialize(model)
        embeddings = ReRanker.generate_embeddings(model, documents)
        ReRanker.store_documents(faiss_index, embeddings)
        query_embedding = ReRanker.generate_embeddings(model, [query])
        distances, indices = ReRanker.queryk(faiss_index, query_embedding, k=k)
        return distances, indices[0]
    










