

import faiss

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






