
from src.vector.bm25 import create_index
from src.data.reader import parse_data
from src.vector.indexer import ReRanker
import pyterrier as pt
import pandas as pd


if not pt.started():
    pt.init()
    

index_folder = "./index"
query = "Identify organizations that participate in international criminal\nactivity, the activity, and, if possible, collaborating organizations\nand the countries involved."
data = parse_data()
docs_df = pd.DataFrame(data["documents"].values())
indexref = create_index(docs_df, index_folder=index_folder)
bm25 = pt.terrier.BatchRetrieve(indexref, wmodel="BM25", num_results=200)
results = bm25.search(query)
print("BM25 Results:")
print(results)

docs_to_rerank = pd.merge(results, docs_df, on="docno", how="left")
docs_to_rerank = docs_to_rerank[["docno", "text"]]
reranked_d, reranked_indices  = ReRanker.rerank(docs_to_rerank["text"].tolist(), query, k=10)
print("Reranked Results:")
print(reranked_d, reranked_indices)


# create initial order of docnos and reranked docnos
initial_order = docs_to_rerank["docno"].tolist()
reranked_order = docs_to_rerank.iloc[reranked_indices]["docno"].tolist()


a = 5



