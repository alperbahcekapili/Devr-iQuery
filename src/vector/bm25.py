import os
import shutil
import pyterrier as pt
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
NLTK_AVAILABLE = True
STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

def clean_text(text):
    """Clean text to ensure it's Java-friendly"""
    if not isinstance(text, str):
        return str(text)
    # Remove control characters and other problematic characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    # Replace problematic characters with spaces
    text = re.sub(r'[\u0000-\u0008\u000B\u000C\u000E-\u001F]', ' ', text)
    return text.strip()

def preprocess_for_bm25(text):
    """
    Preprocess text specifically for BM25 indexing and querying.
    Applies: lowercasing, special char/number removal, stopword removal, stemming
    """
    if not isinstance(text, str):
        text = str(text)
    
    # First apply basic cleaning
    text = clean_text(text)
    
    # Lowercasing
    text = text.lower()
    
    # Remove special characters and numbers, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if NLTK_AVAILABLE and text:
        # Tokenize, remove stopwords, and stem
        words = text.split()
        
        # Remove stopwords
        words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
        
        # Stem words
        if STEMMER:
            words = [STEMMER.stem(word) for word in words]
        
        text = ' '.join(words)
    
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
    
    # Store original text for rerankers and create processed version for BM25
    original_text = clean_text(doc['text'])  # Basic cleaning only
    processed_text = preprocess_for_bm25(doc['text'])  # Full BM25 preprocessing
    
    doc['text'] = processed_text  # BM25 will use this
    doc['original_text'] = original_text  # Store original for rerankers
    
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
    
    # Validate the data with preprocessing
    validated_docs = validate_data(docs_dict)
    print(f"Number of documents after preprocessing and validation: {len(validated_docs)}")
    
    # Create index with validated data
    indexer = pt.IterDictIndexer(index_folder, meta={'docno': 50, 'text': 32768, 'original_text': 32768}, overwrite=True)
    indexref = indexer.index(validated_docs)
    return indexref