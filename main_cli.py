from src.vector.bm25 import create_index, preprocess_for_bm25
from src.data.reader import parse_data
from src.vector.indexer import ReRanker
import pyterrier as pt
import pandas as pd

index_folder = "./index"

# Original query
original_query = "Identify organizations that participate in international criminal\nactivity, the activity, and, if possible, collaborating organizations\nand the countries involved."

# Preprocess query for BM25
bm25_query = preprocess_for_bm25(original_query)

print(f"Original query: {original_query}")
print(f"BM25 preprocessed query: {bm25_query}")

data = parse_data()
docs_df = pd.DataFrame(data["documents"].values())

# Create index (documents will be preprocessed automatically)
indexref = create_index(docs_df, index_folder=index_folder)

# BM25 search using preprocessed query
bm25 = pt.terrier.BatchRetrieve(indexref, wmodel="BM25", num_results=200)
results = bm25.search(bm25_query)  # Use preprocessed query
print("BM25 Results:")
print(results)

# For reranking, use original documents text (not preprocessed)
docs_to_rerank = pd.merge(results, docs_df, on="docno", how="left")

# Use original_text if available (from new preprocessing), otherwise use text
if 'original_text' in docs_to_rerank.columns:
    docs_to_rerank = docs_to_rerank[["docno", "original_text"]].rename(columns={"original_text": "text"})
    print("Using original text for reranking")
else:
    docs_to_rerank = docs_to_rerank[["docno", "text"]]
    print("Using text for reranking")

# Rerank using original query and original document text
reranked_d, reranked_indices = ReRanker.rerank(
    docs_to_rerank["text"].tolist(), 
    original_query,  # Use original query for reranking
    k=10
)
print("Reranked Results:")
print(reranked_d, reranked_indices)

# create initial order of docnos and reranked docnos
initial_order = docs_to_rerank["docno"].tolist()
reranked_order = docs_to_rerank.iloc[reranked_indices]["docno"].tolist()

a = 5