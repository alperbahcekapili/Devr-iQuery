import os
import shutil
import pyterrier as pt

def create_index(data, index_folder="./index"):
    if os.path.exists(index_folder) and os.path.isdir(index_folder):
        shutil.rmtree(index_folder)
    indexer = pt.IterDictIndexer(index_folder, meta={'docno': 20, 'text': 4096})
    indexref = indexer.index(data.to_dict(orient="records"))
    return indexref