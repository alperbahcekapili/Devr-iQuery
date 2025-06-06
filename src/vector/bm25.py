import os
import shutil
import pyterrier as pt
import json
import re

def clean_text(text):
    """Clean text to ensure it's Java-friendly"""
    if not isinstance(text, str):
        return str(text)
    # Remove control characters and other problematic characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    # Replace problematic characters with spaces
    text = re.sub(r'[\u0000-\u0008\u000B\u000C\u000E-\u001F]', ' ', text)
    return text.strip()

def validate_document(doc):
    """Validate a single document"""
    if not isinstance(doc, dict):
        raise ValueError(f"Document must be a dictionary, got {type(doc)}")
    
    if 'docno' not in doc:
        raise ValueError("Document missing 'docno' field")
    if 'text' not in doc:
        raise ValueError("Document missing 'text' field")
    
    # Clean and validate fields
    doc['docno'] = str(doc['docno']).strip()
    doc['text'] = clean_text(doc['text'])
    
    return doc

def validate_data(data):
    """Validate the entire dataset"""
    if not isinstance(data, list):
        raise ValueError(f"Data must be a list of documents, got {type(data)}")
    
    validated_docs = []
    for i, doc in enumerate(data):
        try:
            validated_doc = validate_document(doc)
            validated_docs.append(validated_doc)
        except Exception as e:
            print(f"Error validating document {i}: {str(e)}")
            raise
    
    return validated_docs

def create_index(data, index_folder="./index"):
    if os.path.exists(index_folder) and os.path.isdir(index_folder):
        shutil.rmtree(index_folder)
    
    # Convert DataFrame to dict and validate
    docs_dict = data.to_dict(orient="records")
    print(f"Number of documents before validation: {len(docs_dict)}")
    
    # Validate the data
    validated_docs = validate_data(docs_dict)
    print(f"Number of documents after validation: {len(validated_docs)}")
    
    # Create index with validated data
    indexer = pt.IterDictIndexer(index_folder, meta={'docno': 50, 'text': 32768}, overwrite=True)
    indexref = indexer.index(validated_docs)
    return indexref