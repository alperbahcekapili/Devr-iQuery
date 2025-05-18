import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class FineGrainedReRanker:
    def __init__(self, model_name='intfloat/e5-mistral-7b-instruct'):
        """
        Initializes the FineGrainedReRanker.

        :param model_name: The name of the sentence-transformer model to use.
        """
        self._model = None
        self.model_name = model_name

    @property
    def model(self):
        """
        Lazy loads the SentenceTransformer model.
        """
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_token_embeddings_with_mapping(self, texts_list):
        """
        Generates token embeddings for a list of texts and creates a mapping
        from a concatenated token index back to its original document index.
        Ensures embeddings are float32 NumPy arrays.

        :param texts_list: A list of document strings.
        :return: A tuple containing:
                 - concatenated_embeddings (np.ndarray): A 2D array of all token embeddings (float32).
                 - token_to_doc_mapping (list): A list where the i-th element is the
                                                document index for the i-th token in
                                                concatenated_embeddings.
        """
        raw_token_embeds_per_doc = self.model.encode(
            texts_list,
            convert_to_numpy=True, # Attempts to convert to NumPy
            output_value="token_embeddings"
        )

        processed_token_embeddings_list = []
        token_to_doc_mapping = []

        for doc_idx, doc_token_embeds in enumerate(raw_token_embeds_per_doc):
            if doc_token_embeds is not None:
                # Ensure it's a NumPy array
                if not isinstance(doc_token_embeds, np.ndarray):
                    if hasattr(doc_token_embeds, 'cpu') and hasattr(doc_token_embeds, 'numpy'): # PyTorch Tensor
                        doc_token_embeds = doc_token_embeds.cpu().numpy()
                    else: # Other types
                        doc_token_embeds = np.array(doc_token_embeds)
                
                # Ensure dtype is float32
                if doc_token_embeds.dtype != np.float32:
                    doc_token_embeds = doc_token_embeds.astype(np.float32)

                if doc_token_embeds.ndim == 2 and doc_token_embeds.shape[0] > 0:
                    processed_token_embeddings_list.append(doc_token_embeds)
                    token_to_doc_mapping.extend([doc_idx] * doc_token_embeds.shape[0])

        if not processed_token_embeddings_list:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return np.array([]).reshape(0, embedding_dim).astype(np.float32), []

        concatenated_embeddings = np.concatenate(processed_token_embeddings_list, axis=0)
        # Ensure final concatenated array is float32 (should be if components are)
        if concatenated_embeddings.dtype != np.float32:
            concatenated_embeddings = concatenated_embeddings.astype(np.float32)
            
        return concatenated_embeddings, token_to_doc_mapping


    def rerank(self, documents, query, k=10):
        """
        Reranks a list of documents based on fine-grained token-level interactions
        with a query. Scores are calculated using a MaxSim approach.

        :param documents: A list of document strings to be reranked.
        :param query: The query string.
        :param k: The number of top documents to return.
        :return: A tuple (scores, indices):
                 - scores (list): A list of relevance scores for the top-k documents. Higher is better.
                 - indices (list): A list of original indices of the top-k documents from the input list.
        """
        if not documents:
            return [], []

        # 1. Generate token embeddings for the query
        query_token_embeds_list = self.model.encode(
            [query],
            convert_to_numpy=True,
            output_value="token_embeddings"
        )

        if not query_token_embeds_list or query_token_embeds_list[0] is None or query_token_embeds_list[0].shape[0] == 0:
            num_return = min(k, len(documents))
            return [0.0] * num_return, list(range(num_return))

        query_tokens = query_token_embeds_list[0]

        # Explicitly ensure query_tokens is a NumPy array and float32
        if not isinstance(query_tokens, np.ndarray):
            if hasattr(query_tokens, 'cpu') and hasattr(query_tokens, 'numpy'): # PyTorch Tensor
                query_tokens = query_tokens.cpu().numpy()
            else: # Fallback for other types
                query_tokens = np.array(query_tokens)
        
        if query_tokens.dtype != np.float32:
            query_tokens = query_tokens.astype(np.float32)

        if query_tokens.ndim == 1:
            query_tokens = np.expand_dims(query_tokens, axis=0)
        
        # Normalize query token embeddings
        norms = np.linalg.norm(query_tokens, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9 # Avoid division by zero for zero-vectors
        query_tokens_norm = (query_tokens / norms).astype(np.float32) # Ensure float32
        
        embedding_dimension = query_tokens_norm.shape[1]
        num_query_tokens = query_tokens_norm.shape[0]

        # 2. Generate token embeddings for all provided documents
        doc_texts = [doc_content for doc_content in documents]
        all_doc_token_embeddings, token_to_doc_mapping = self._get_token_embeddings_with_mapping(doc_texts)

        if all_doc_token_embeddings.shape[0] == 0:
            num_return = min(k, len(documents))
            return [0.0] * num_return, list(range(num_return))

        # Normalize document token embeddings (all_doc_token_embeddings is already float32 from helper)
        doc_norms = np.linalg.norm(all_doc_token_embeddings, axis=1, keepdims=True)
        doc_norms[doc_norms == 0] = 1e-9
        all_doc_token_embeddings_norm = (all_doc_token_embeddings / doc_norms).astype(np.float32)

        # Ensure C-contiguity for Faiss
        if not all_doc_token_embeddings_norm.flags['C_CONTIGUOUS']:
            all_doc_token_embeddings_norm = np.ascontiguousarray(all_doc_token_embeddings_norm, dtype=np.float32)
        # (Type should be float32 already, but double check)
        elif all_doc_token_embeddings_norm.dtype != np.float32:
            all_doc_token_embeddings_norm = all_doc_token_embeddings_norm.astype(np.float32)


        # 3. Build a Faiss index for all document tokens
        index = faiss.IndexFlatIP(embedding_dimension)
        index.add(all_doc_token_embeddings_norm)

        # 4. For each query token, search the Faiss index
        k_search_tokens = min(10, index.ntotal)
        if k_search_tokens == 0:
            num_return = min(k, len(documents))
            return [0.0] * num_return, list(range(num_return))

        # Ensure query_tokens_norm is C-contiguous and float32 for Faiss search
        if not query_tokens_norm.flags['C_CONTIGUOUS']:
            query_tokens_norm = np.ascontiguousarray(query_tokens_norm, dtype=np.float32)
        # (Type should be float32 already, but double check)
        elif query_tokens_norm.dtype != np.float32:
            query_tokens_norm = query_tokens_norm.astype(np.float32)
        
        sim_matrix_qt_dt, indices_matrix_dt = index.search(query_tokens_norm, k_search_tokens)

        # 5. Aggregate scores at the document level (MaxSim style)
        max_sim_scores_doc_qtoken = np.full((len(documents), num_query_tokens), -np.inf, dtype=float)

        for q_idx in range(num_query_tokens):
            query_token_similarities = sim_matrix_qt_dt[q_idx]
            doc_token_global_indices = indices_matrix_dt[q_idx]

            for i in range(len(query_token_similarities)):
                retrieved_doc_token_idx = doc_token_global_indices[i]
                if retrieved_doc_token_idx == -1:
                    continue
                similarity_score = query_token_similarities[i]
                original_document_index = token_to_doc_mapping[retrieved_doc_token_idx]
                if similarity_score > max_sim_scores_doc_qtoken[original_document_index, q_idx]:
                    max_sim_scores_doc_qtoken[original_document_index, q_idx] = similarity_score
        
        max_sim_scores_doc_qtoken[max_sim_scores_doc_qtoken == -np.inf] = 0.0
        final_document_scores = np.sum(max_sim_scores_doc_qtoken, axis=1)

        # 6. Sort documents by their aggregated scores
        scored_doc_tuples = sorted(
            [(final_document_scores[i], i) for i in range(len(documents))],
            key=lambda x: x[0],
            reverse=True
        )
        
        top_k_scores = [score for score, idx in scored_doc_tuples[:k]]
        top_k_indices = [idx for score, idx in scored_doc_tuples[:k]]
        
        return top_k_scores, top_k_indices

# Example Usage (optional, for testing):
if __name__ == '__main__':
    reranker = FineGrainedReRanker()

    docs_to_rerank = [
        "The sky is blue and vast.",
        "An apple a day keeps the doctor away.",
        "Blue berries are tasty and good for health.",
        "Artificial intelligence is a fascinating field.",
        "The ocean blue is deep and mysterious."
    ]
    query_text = "healthy blue fruits"
    
    top_k = 3
    scores, indices = reranker.rerank(docs_to_rerank, query_text, k=top_k)
    
    print(f"Query: {query_text}\n")
    print(f"Top {top_k} Reranked Documents:")
    for i in range(len(scores)):
        doc_idx = indices[i]
        print(f"  Score: {scores[i]:.4f}, Index: {doc_idx}, Document: \"{docs_to_rerank[doc_idx]}\"")

    docs_with_empty = [
        "The sky is blue.",
        "", 
        "Blue berries are tasty."
    ]
    query_text_2 = "blue things"
    scores2, indices2 = reranker.rerank(docs_with_empty, query_text_2, k=2)
    print(f"\nQuery: {query_text_2}\n")
    print(f"Top {len(scores2)} Reranked Documents (with empty doc in list):")
    for i in range(len(scores2)):
        doc_idx = indices2[i]
        print(f"  Score: {scores2[i]:.4f}, Index: {doc_idx}, Document: \"{docs_with_empty[doc_idx]}\"")

    docs_for_empty_query = ["doc1", "doc2"]
    empty_query = "" # "" query
    scores3, indices3 = reranker.rerank(docs_for_empty_query, empty_query, k=2)
    print(f"\nQuery: '{empty_query}' (empty)\n") # Corrected print for empty query
    print(f"Top {len(scores3)} Reranked Documents (empty query):")
    for i in range(len(scores3)):
        doc_idx = indices3[i]
        print(f"  Score: {scores3[i]:.4f}, Index: {doc_idx}, Document: \"{docs_for_empty_query[doc_idx]}\"")