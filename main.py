
from src.data.reader import parse_data
import json
import pandas as pd
import pyterrier as pt

if not pt.started():
    pt.init()
    
# Example usage:
folder = 'data/ft/all'
# parsed_data = parse_documents(folder)

topic_path = "/home/alpfischer/Devr-iQuery/data/query-relJudgments/q-topics-org-SET2.txt"
# topis = parse_topics(topic_path)
# qrels = parse_qrel_judgements("/home/alpfischer/Devr-iQuery/data/query-relJudgments/qrel_301-350_complete.txt")
data = parse_data()

docs_df = pd.DataFrame(data["documents"].values())
# docs = pd.DataFrame([
#     {"docno": "doc1", "text": "PyTerrier is a great toolkit for IR."},
#     {"docno": "doc2", "text": "BM25 is a ranking function used in IR."},
#     {"docno": "doc3", "text": "This document is about information retrieval."}
# ])

def create_index(data):
    

    indexer = pt.IterDictIndexer('./my_index')

    # Thisws creates an index with the default settings (including BM25-compatible indexing)
    indexref = indexer.index(docs_df.to_dict(orient="records"))

    return indexref

indexref = create_index(docs_df)
bm25 = pt.terrier.Retriever(indexref, wmodel="BM25")

results = bm25.search("information retrieval")
print(results[['docno', 'score']])