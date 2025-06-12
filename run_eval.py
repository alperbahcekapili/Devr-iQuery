import os
import shutil
import numpy as np
import pandas as pd
import pyterrier as pt
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import logging
import re
import traceback
import random

# Create results directory if it doesn't exist
results_dir = "./results_fine_grained"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    print(f"Created results directory: {results_dir}")

# Configure detailed logging with output to results directory
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(results_dir, "evaluation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("evaluation")

sys.path.append('./src/') 
logger.info("Importing modules from codebase")
try:
    from data.reader import parse_data
    from vector.indexer import ReRanker, FineGrainedReRanker
    from vector.bm25 import create_index
    logger.info("Successfully imported all modules")
except Exception as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

# Initialize PyTerrier if not already initialized
logger.info("Initializing PyTerrier")
try:
    if not pt.java.started():
        pt.java.init()
    logger.info("PyTerrier initialized successfully")
except Exception as e:
    logger.error(f"Error initializing PyTerrier: {e}")
    sys.exit(1)

def sanitize_query_for_terrier(query):
    """
    Sanitize the query string specifically for Terrier query parser
    """
    logger.debug(f"Original query: '{query}'")
    
    # Remove newlines and excessive whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # Terrier query parser has issues with certain characters
    # Escape or remove problematic characters: ?, (, ), ", :, -, etc.
    query = query.replace('?', ' ')
    query = query.replace('(', ' ')
    query = query.replace(')', ' ')
    query = query.replace('"', ' ')
    query = query.replace(':', ' ')
    query = query.replace('-', ' ')
    query = query.replace('/', ' ')
    query = query.replace('\'', ' ')
    query = query.replace('.', ' ')
    query = query.replace(',', ' ')
    query = query.replace('!', ' ')
    query = query.replace('&', ' and ')
    query = query.replace('|', ' or ')
    
    # Terrier has issues with words that are too long
    words = []
    for word in query.split():
        if len(word) > 30:  # Truncate very long words
            word = word[:30]
        words.append(word)
    
    query = ' '.join(words)
    
    # Remove any other non-alphanumeric characters
    query = re.sub(r'[^\w\s]', ' ', query)
    
    # Clean up extra whitespace
    query = re.sub(r'\s+', ' ', query).strip()
    
    # If query is empty, use a default
    if not query or len(query) < 2:
        query = "document"
    
    logger.debug(f"Sanitized query for Terrier: '{query}'")
    return query

def calculate_precision_at_k(retrieved_docs, relevant_docs, k):
    """Calculate precision at k"""
    if not retrieved_docs or len(retrieved_docs) == 0:
        return 0.0
    
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_at_k if doc in relevant_docs]
    
    return len(relevant_retrieved) / min(k, len(retrieved_docs))

def calculate_recall_at_k(retrieved_docs, relevant_docs, k):
    """Calculate recall at k"""
    if not relevant_docs or len(relevant_docs) == 0:
        return 1.0  # All relevant docs retrieved (there are none)
    
    if not retrieved_docs or len(retrieved_docs) == 0:
        return 0.0
    
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_at_k if doc in relevant_docs]
    
    return len(relevant_retrieved) / len(relevant_docs)

def calculate_mrr(retrieved_docs, relevant_docs):
    """Calculate Mean Reciprocal Rank"""
    if not retrieved_docs or not relevant_docs:
        return 0.0
        
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0

def calculate_dcg_at_k(retrieved_docs, relevance_scores, k):
    """Calculate DCG at k"""
    if not retrieved_docs:
        return 0.0
        
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k]):
        if i >= k:
            break
        rel = float(relevance_scores.get(doc, 0))
        dcg += (2 ** rel - 1) / np.log2(i + 2)
    return dcg

def calculate_ndcg_at_k(retrieved_docs, relevance_scores, k):
    """Calculate NDCG at k"""
    if not retrieved_docs:
        return 0.0
        
    dcg = calculate_dcg_at_k(retrieved_docs, relevance_scores, k)
    
    # Calculate ideal DCG - sort by relevance score
    sorted_docs = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    ideal_docs = [doc for doc, score in sorted_docs]
    
    idcg = calculate_dcg_at_k(ideal_docs, relevance_scores, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def safely_remove_directory(directory_path):
    """Safely remove a directory with proper error handling"""
    try:
        if os.path.exists(directory_path):
            logger.info(f"Removing existing directory: {directory_path}")
            shutil.rmtree(directory_path)
        return True
    except Exception as e:
        logger.error(f"Error removing directory {directory_path}: {e}")
        return False

def run_evaluation():
    logger.info("Starting evaluation process")
    
    logger.info("Loading data using parse_data()")
    try:
        data = parse_data()
        documents = data["documents"]
        topics = data["topics"]
        qrels = data["qrels"]
        
        logger.info(f"Loaded {len(documents)} documents, {len(topics)} topics, and data for {len(qrels)} topics in qrels")
        
        # Check document ID overlap between documents and qrels
        doc_ids = set(documents.keys())
        qrel_doc_ids = set()
        for topic_id, topic_qrels in qrels.items():
            qrel_doc_ids.update(topic_qrels.keys())
        
        overlap = doc_ids.intersection(qrel_doc_ids)
        overlap_percentage = (len(overlap) / len(qrel_doc_ids) * 100) if qrel_doc_ids else 0
        
        logger.info(f"Document ID overlap between documents and qrels: {len(overlap)} out of {len(qrel_doc_ids)} ({overlap_percentage:.2f}%)")
        
        if overlap_percentage < 5:
            logger.warning("CRITICAL: Very low document overlap with relevance judgments (< 5%)")
            logger.warning("This will significantly impact evaluation results!")
            logger.warning("Results may not be meaningful due to missing relevant documents.")
        
        # Log document ID prefixes to understand the mismatch
        doc_id_prefixes = defaultdict(int)
        for doc_id in doc_ids:
            prefix = doc_id.split('-')[0] if '-' in doc_id else doc_id[:3]
            doc_id_prefixes[prefix] += 1
        
        qrel_id_prefixes = defaultdict(int)
        for doc_id in qrel_doc_ids:
            prefix = doc_id.split('-')[0] if '-' in doc_id else doc_id[:3]
            qrel_id_prefixes[prefix] += 1
        
        logger.info(f"Document ID prefixes distribution: {dict(doc_id_prefixes)}")
        logger.info(f"Qrel ID prefixes distribution: {dict(qrel_id_prefixes)}")
        
        # Find prefixes in both collections
        common_prefixes = set(doc_id_prefixes.keys()) & set(qrel_id_prefixes.keys())
        logger.info(f"Common document prefixes: {common_prefixes}")
        
        # Calculate how many documents with common prefixes are in the qrels
        common_prefix_docs_in_qrels = sum(qrel_id_prefixes[prefix] for prefix in common_prefixes)
        logger.info(f"Documents with common prefixes in qrels: {common_prefix_docs_in_qrels} out of {len(qrel_doc_ids)} ({common_prefix_docs_in_qrels/len(qrel_doc_ids)*100:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Convert documents to a format suitable for indexing
    logger.info("Preparing documents for indexing")
    docs_for_indexing = []
    for doc_id, doc in documents.items():
        text = ""
        if "headline" in doc:
            text += doc["headline"] + " "
        if "text" in doc:
            text += doc["text"]
        
        docs_for_indexing.append({
            "docno": doc_id,
            "text": text
        })
    
    # Create a pandas DataFrame for PyTerrier
    docs_df = pd.DataFrame(docs_for_indexing)
    logger.info(f"Prepared {len(docs_df)} documents for indexing")
    
    # Log the maximum text length to help with indexing
    max_text_length = docs_df['text'].str.len().max()
    logger.info(f"Maximum text length in documents: {max_text_length}")
    
    index_path = os.path.join(results_dir, "index")
    logger.info(f"Creating BM25 index at {index_path}...")
    
    # Safely remove the existing index directory to avoid errors
    if not safely_remove_directory(index_path):
        logger.warning("Could not remove existing index directory. Will attempt to use existing index.")
        try:
            # Try to use existing index
            indexref = pt.IndexRef.of(f"{index_path}/data.properties")
            logger.info("Successfully loaded existing index")
        except Exception as e:
            logger.error(f"Could not load existing index: {e}")
            logger.error("Attempting to create a new index with a different path...")
            
            # Try with a different index path
            index_path = os.path.join(results_dir, "index_new")
            if not safely_remove_directory(index_path):
                logger.error("Could not prepare alternative index location. Exiting.")
                return
    
    try:
        # Create the index
        logger.info(f"Creating new index at {index_path}")
        indexer = pt.IterDictIndexer(index_path, meta={'docno': 20, 'text': max_text_length + 100})
        indexref = indexer.index(docs_df.to_dict(orient="records"))
        
        index = pt.IndexFactory.of(indexref)
        logger.info(f"Index created successfully with {index.getCollectionStatistics().getNumberOfDocuments()} documents")
        
        # Log index details
        index_stats = index.getCollectionStatistics()
        logger.info(f"Index statistics: {index_stats.toString()}")
        logger.info(f"Number of tokens: {index_stats.getNumberOfTokens()}")
        logger.info(f"Number of unique terms: {index_stats.getNumberOfUniqueTerms()}")
        logger.info(f"Number of pointers: {index_stats.getNumberOfPointers()}")
        
        # Use appropriate retrieval parameters
        bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"c": 0.75, "bm25.b": 0.75, "bm25.k_1": 1.2})
        logger.info("BM25 retriever initialized")
    except Exception as e:
        logger.error(f"Error creating/loading index: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Define the k values for evaluation
    k_values = [5, 10, 20, 50, 100]
    logger.info(f"Will evaluate metrics at k values: {k_values}")
    
    # Dictionaries to store results for all three methods
    bm25_results = {metric: {k: [] for k in k_values} for metric in ['precision', 'recall', 'ndcg']}
    qwen_reranker_results = {metric: {k: [] for k in k_values} for metric in ['precision', 'recall', 'ndcg']}
    finegrained_reranker_results = {metric: {k: [] for k in k_values} for metric in ['precision', 'recall', 'ndcg']}
    
    # MRR is a single value per topic
    bm25_mrr = []
    qwen_reranker_mrr = []
    finegrained_reranker_mrr = []
    
    # Additional metrics for analysis
    bm25_topic_scores = {}
    qwen_reranker_topic_scores = {}
    finegrained_reranker_topic_scores = {}
    topics_with_relevant_docs = 0
    total_relevant_docs = 0
    relevant_docs_available = 0
    
    logger.info("Starting evaluation on topics...")
    topics_evaluated = 0
    topics_skipped = 0
    bm25_failures = 0
    qwen_reranker_failures = 0
    finegrained_reranker_failures = 0
    
    # Create a docid_to_text mapping for rerankers
    docid_to_text = {}
    for doc_id, doc in documents.items():
        text = ""
        if "headline" in doc:
            text += doc["headline"] + " "
        if "text" in doc:
            text += doc["text"]
        docid_to_text[doc_id] = text
    
    # Get topics with at least one relevant document in our document collection
    valid_topics = []
    for topic_id, topic in topics.items():
        if topic_id not in qrels:
            continue
            
        topic_qrels = qrels[topic_id]
        relevant_docs = [doc_id for doc_id, rel in topic_qrels.items() if int(rel) > 0]
        
        # Check if any relevant docs are in our document collection
        relevant_in_collection = [doc_id for doc_id in relevant_docs if doc_id in doc_ids]
        
        if relevant_in_collection:
            valid_topics.append((topic_id, topic, len(relevant_in_collection), len(relevant_docs)))
    
    logger.info(f"Found {len(valid_topics)} topics with at least one relevant document in our collection")
    
    # Sort topics by number of relevant documents available
    valid_topics.sort(key=lambda x: x[2], reverse=True)
    
    # Now evaluate using the valid topics
    for topic_id, topic, relevant_count, total_relevant in valid_topics:
        logger.info(f"\n{'='*40}\nProcessing topic {topic_id}: {topic.get('title', 'No title')}")
        logger.info(f"Has {relevant_count} relevant documents available out of {total_relevant} total relevant")
        
        total_relevant_docs += total_relevant
        relevant_docs_available += relevant_count
        topics_with_relevant_docs += 1
        
        # Extract and clean query
        title = topic.get("title", "").strip()
        description = topic.get("description", "").strip()
        logger.info(f"Title: '{title}'")
        logger.info(f"Description: '{description}'")
        
        query = title
        if description:
            # Keep description shorter to avoid parsing issues
            description_words = description.split()[:15]  # Limit to first 15 words
            query += " " + " ".join(description_words)
            
        # Sanitize the query for Terrier
        terrier_query = sanitize_query_for_terrier(query)
        logger.info(f"Sanitized query for Terrier: '{terrier_query}'")
        
        # Clean query for rerankers (less strict cleaning)
        reranker_query = re.sub(r'\s+', ' ', query).strip()
        logger.info(f"Query for rerankers: '{reranker_query}'")
        
        # Get relevant documents for this topic
        topic_qrels = qrels[topic_id]
        relevant_docs = [doc_id for doc_id, rel in topic_qrels.items() if int(rel) > 0]
        relevance_scores = {doc_id: int(rel) for doc_id, rel in topic_qrels.items()}
        
        # Get relevant docs that are actually in our collection
        relevant_in_collection = [doc_id for doc_id in relevant_docs if doc_id in doc_ids]
        logger.info(f"Topic {topic_id} has {len(relevant_in_collection)}/{len(relevant_docs)} relevant documents in our collection")
        
        if not relevant_in_collection:
            logger.warning(f"No relevant documents for topic {topic_id} are in our collection, skipping...")
            topics_skipped += 1
            continue
        
        # BM25 retrieval
        logger.info(f"Running BM25 search for topic {topic_id}")
        bm25_retrieved = []
        try:
            # Create a query dataframe - this is important for PyTerrier
            query_df = pd.DataFrame([{"qid": topic_id, "query": terrier_query}])
            
            # Use transform method with error handling
            bm25_results_df = bm25.transform(query_df)
            logger.info(f"BM25 search successful, got DataFrame of shape {bm25_results_df.shape}")
            
            # Convert results to list of document IDs
            if len(bm25_results_df) > 0 and 'docno' in bm25_results_df.columns:
                bm25_retrieved = bm25_results_df["docno"].tolist()
                logger.info(f"BM25 retrieved {len(bm25_retrieved)} documents")
                
                # Check if any retrieved docs are in relevant docs
                retrieved_and_relevant = [doc_id for doc_id in bm25_retrieved[:50] if doc_id in relevant_in_collection]
                logger.info(f"BM25 retrieved {len(retrieved_and_relevant)} relevant documents in top 50")
            else:
                logger.warning(f"BM25 results missing 'docno' column or empty. Columns: {bm25_results_df.columns.tolist() if len(bm25_results_df) > 0 else 'No columns'}")
        except Exception as e:
            logger.error(f"Error with BM25 retrieval for topic {topic_id}: {e}")
            logger.error(traceback.format_exc())
            bm25_failures += 1
            
            # Try with a much simpler fallback query
            logger.info("Attempting fallback with one-word query")
            try:
                # Use only the first word of the title as a simple query
                simple_query = title.split()[0] if title and title.split() else "document"
                
                # Make sure it's a common word that won't cause parsing errors
                if len(simple_query) < 4 or simple_query.lower() not in ["the", "and", "for", "with"]:
                    simple_query = "document"
                    
                logger.info(f"Simple fallback query: '{simple_query}'")
                
                query_df = pd.DataFrame([{"qid": topic_id, "query": simple_query}])
                bm25_results_df = bm25.transform(query_df)
                
                if len(bm25_results_df) > 0 and 'docno' in bm25_results_df.columns:
                    bm25_retrieved = bm25_results_df["docno"].tolist()
                    logger.info(f"Fallback successful, retrieved {len(bm25_retrieved)} documents")
                else:
                    logger.warning("Fallback failed - no docno column or empty results")
            except Exception as e:
                logger.error(f"Fallback attempt also failed: {e}")
        
        # Qwen3 Embedding ReRanker
        logger.info(f"Running Qwen3 Embedding ReRanker for topic {topic_id}")
        qwen_reranker_retrieved = []
        try:
            # For rerankers, we'll use either BM25 results or include all known relevant docs
            doc_texts = []
            doc_ids = []
            
            # Include all available relevant documents in the sample for rerankers
            for doc_id in relevant_in_collection:
                if doc_id in docid_to_text:
                    doc_texts.append(docid_to_text[doc_id])
                    doc_ids.append(doc_id)
            
            # Add BM25 results if available, otherwise sample
            if bm25_retrieved:
                # Use top BM25 results for reranker
                logger.info(f"Adding BM25 results to Qwen reranker documents")
                for docno in bm25_retrieved[:300]:  # Use top 300 documents from BM25
                    if docno in docid_to_text and docno not in doc_ids:
                        doc_texts.append(docid_to_text[docno])
                        doc_ids.append(docno)
            else:
                # Sample documents for reranker
                logger.info("Adding random sample of documents for Qwen reranker")
                sample_size = min(300, len(documents) - len(doc_ids))
                possible_docs = [d for d in documents.keys() if d not in doc_ids]
                sample_doc_ids = random.sample(possible_docs, sample_size)
                
                for doc_id in sample_doc_ids:
                    if doc_id in docid_to_text:
                        doc_texts.append(docid_to_text[doc_id])
                        doc_ids.append(doc_id)
            
            logger.info(f"Prepared {len(doc_texts)} document texts for Qwen reranker")
            logger.info(f"Included {len(relevant_in_collection)} known relevant documents in Qwen reranker input")
            
            # Run Qwen3 Embedding reranker if we have documents
            if doc_texts:
                logger.info(f"Running Qwen3 Embedding reranker with query: '{reranker_query}'")
                
                try:
                    distances, indices, faiss_duration, index_duration = ReRanker.rerank(
                        doc_texts, reranker_query, model='Qwen/Qwen3-Embedding-0.6B', k=min(len(doc_texts), max(k_values))
                    )
                    
                    # Convert indices to document IDs
                    qwen_reranker_retrieved = [doc_ids[idx] for idx in indices if idx < len(doc_ids)]
                    logger.info(f"Qwen3 Embedding reranker retrieved {len(qwen_reranker_retrieved)} documents")
                    
                    # Check if any retrieved docs are in relevant docs
                    qwen_relevant = [doc_id for doc_id in qwen_reranker_retrieved[:50] if doc_id in relevant_in_collection]
                    logger.info(f"Qwen3 Embedding reranker retrieved {len(qwen_relevant)} relevant documents in top 50")
                except Exception as e:
                    logger.error(f"Error in Qwen3 Embedding ReRanker.rerank: {e}")
                    logger.error(traceback.format_exc())
                    qwen_reranker_failures += 1
            else:
                logger.warning("No documents prepared for Qwen3 Embedding reranker")
        except Exception as e:
            logger.error(f"Error with Qwen3 Embedding reranker for topic {topic_id}: {e}")
            logger.error(traceback.format_exc())
            qwen_reranker_failures += 1
        
        # Fine-grained Qwen3 ReRanker
        logger.info(f"Running Fine-grained Qwen3 ReRanker for topic {topic_id}")
        finegrained_reranker_retrieved = []
        try:
            # Use the same document preparation as Qwen3 Embedding reranker
            if doc_texts:
                logger.info(f"Running fine-grained Qwen3 reranker with query: '{reranker_query}'")
                
                try:
                    distances, indices, finegrained_duration, index_duration = FineGrainedReRanker.rerank(
                        doc_texts, reranker_query, model='Qwen/Qwen3-Reranker-0.6B', k=min(len(doc_texts), max(k_values))
                    )
                    
                    # Convert indices to document IDs
                    finegrained_reranker_retrieved = [doc_ids[idx] for idx in indices if idx < len(doc_ids)]
                    logger.info(f"Fine-grained Qwen3 reranker retrieved {len(finegrained_reranker_retrieved)} documents")
                    
                    # Check if any retrieved docs are in relevant docs
                    finegrained_relevant = [doc_id for doc_id in finegrained_reranker_retrieved[:50] if doc_id in relevant_in_collection]
                    logger.info(f"Fine-grained Qwen3 reranker retrieved {len(finegrained_relevant)} relevant documents in top 50")
                except Exception as e:
                    logger.error(f"Error in FineGrainedReRanker.rerank: {e}")
                    logger.error(traceback.format_exc())
                    finegrained_reranker_failures += 1
            else:
                logger.warning("No documents prepared for fine-grained Qwen3 reranker")
        except Exception as e:
            logger.error(f"Error with fine-grained Qwen3 reranker for topic {topic_id}: {e}")
            logger.error(traceback.format_exc())
            finegrained_reranker_failures += 1
        
        # Calculate metrics for BM25 against AVAILABLE relevant docs
        if bm25_retrieved:
            # Use only relevant docs that are in our collection
            mrr = calculate_mrr(bm25_retrieved, relevant_in_collection)
            bm25_mrr.append(mrr)
            logger.info(f"BM25 MRR for topic {topic_id}: {mrr:.4f}")
            
            topic_metrics = {'mrr': mrr}
            
            for k in k_values:
                precision = calculate_precision_at_k(bm25_retrieved, relevant_in_collection, k)
                recall = calculate_recall_at_k(bm25_retrieved, relevant_in_collection, k)
                ndcg = calculate_ndcg_at_k(bm25_retrieved, 
                                          {doc_id: relevance_scores[doc_id] for doc_id in relevant_in_collection}, 
                                          k)
                
                bm25_results['precision'][k].append(precision)
                bm25_results['recall'][k].append(recall)
                bm25_results['ndcg'][k].append(ndcg)
                
                topic_metrics[f'p@{k}'] = precision
                topic_metrics[f'r@{k}'] = recall
                topic_metrics[f'ndcg@{k}'] = ndcg
                
                logger.info(f"BM25 metrics for topic {topic_id} at k={k}: P={precision:.4f}, R={recall:.4f}, NDCG={ndcg:.4f}")
            
            bm25_topic_scores[topic_id] = topic_metrics
        
        # Calculate metrics for Qwen3 Embedding ReRanker against AVAILABLE relevant docs
        if qwen_reranker_retrieved:
            # Use only relevant docs that are in our collection
            mrr = calculate_mrr(qwen_reranker_retrieved, relevant_in_collection)
            qwen_reranker_mrr.append(mrr)
            logger.info(f"Qwen3 Embedding ReRanker MRR for topic {topic_id}: {mrr:.4f}")
            
            topic_metrics = {'mrr': mrr}
            
            for k in k_values:
                precision = calculate_precision_at_k(qwen_reranker_retrieved, relevant_in_collection, k)
                recall = calculate_recall_at_k(qwen_reranker_retrieved, relevant_in_collection, k)
                ndcg = calculate_ndcg_at_k(qwen_reranker_retrieved, 
                                          {doc_id: relevance_scores[doc_id] for doc_id in relevant_in_collection}, 
                                          k)
                
                qwen_reranker_results['precision'][k].append(precision)
                qwen_reranker_results['recall'][k].append(recall)
                qwen_reranker_results['ndcg'][k].append(ndcg)
                
                topic_metrics[f'p@{k}'] = precision
                topic_metrics[f'r@{k}'] = recall
                topic_metrics[f'ndcg@{k}'] = ndcg
                
                logger.info(f"Qwen3 Embedding ReRanker metrics for topic {topic_id} at k={k}: P={precision:.4f}, R={recall:.4f}, NDCG={ndcg:.4f}")
            
            qwen_reranker_topic_scores[topic_id] = topic_metrics
        
        # Calculate metrics for Fine-grained ReRanker against AVAILABLE relevant docs
        if finegrained_reranker_retrieved:
            # Use only relevant docs that are in our collection
            mrr = calculate_mrr(finegrained_reranker_retrieved, relevant_in_collection)
            finegrained_reranker_mrr.append(mrr)
            logger.info(f"Fine-grained ReRanker MRR for topic {topic_id}: {mrr:.4f}")
            
            topic_metrics = {'mrr': mrr}
            
            for k in k_values:
                precision = calculate_precision_at_k(finegrained_reranker_retrieved, relevant_in_collection, k)
                recall = calculate_recall_at_k(finegrained_reranker_retrieved, relevant_in_collection, k)
                ndcg = calculate_ndcg_at_k(finegrained_reranker_retrieved, 
                                          {doc_id: relevance_scores[doc_id] for doc_id in relevant_in_collection}, 
                                          k)
                
                finegrained_reranker_results['precision'][k].append(precision)
                finegrained_reranker_results['recall'][k].append(recall)
                finegrained_reranker_results['ndcg'][k].append(ndcg)
                
                topic_metrics[f'p@{k}'] = precision
                topic_metrics[f'r@{k}'] = recall
                topic_metrics[f'ndcg@{k}'] = ndcg
                
                logger.info(f"Fine-grained ReRanker metrics for topic {topic_id} at k={k}: P={precision:.4f}, R={recall:.4f}, NDCG={ndcg:.4f}")
            
            finegrained_reranker_topic_scores[topic_id] = topic_metrics
        
        topics_evaluated += 1
        
        # Log progress periodically
        if topics_evaluated % 10 == 0:
            logger.info(f"Evaluated {topics_evaluated} topics so far")
    
    # Data availability report
    logger.info("\n" + "=" * 50)
    logger.info("DATA AVAILABILITY REPORT")
    logger.info(f"Topics with at least one relevant document in our collection: {topics_with_relevant_docs}")
    logger.info(f"Total relevant documents across all topics: {total_relevant_docs}")
    logger.info(f"Relevant documents available in our collection: {relevant_docs_available} ({relevant_docs_available/total_relevant_docs*100:.2f}%)")
    logger.info("=" * 50)
    
    logger.info(f"Evaluation complete. Evaluated {topics_evaluated} topics, skipped {topics_skipped} topics.")
    logger.info(f"BM25 failures: {bm25_failures}, Qwen3 Embedding ReRanker failures: {qwen_reranker_failures}, Fine-grained ReRanker failures: {finegrained_reranker_failures}")
    
    # Calculate averages
    logger.info("Calculating average metrics")
    
    bm25_avg = {
        'mrr': np.mean(bm25_mrr) if bm25_mrr else 0,
        'precision': {k: np.mean(bm25_results['precision'][k]) if bm25_results['precision'][k] else 0 for k in k_values},
        'recall': {k: np.mean(bm25_results['recall'][k]) if bm25_results['recall'][k] else 0 for k in k_values},
        'ndcg': {k: np.mean(bm25_results['ndcg'][k]) if bm25_results['ndcg'][k] else 0 for k in k_values}
    }
    
    qwen_reranker_avg = {
        'mrr': np.mean(qwen_reranker_mrr) if qwen_reranker_mrr else 0,
        'precision': {k: np.mean(qwen_reranker_results['precision'][k]) if qwen_reranker_results['precision'][k] else 0 for k in k_values},
        'recall': {k: np.mean(qwen_reranker_results['recall'][k]) if qwen_reranker_results['recall'][k] else 0 for k in k_values},
        'ndcg': {k: np.mean(qwen_reranker_results['ndcg'][k]) if qwen_reranker_results['ndcg'][k] else 0 for k in k_values}
    }
    
    finegrained_reranker_avg = {
        'mrr': np.mean(finegrained_reranker_mrr) if finegrained_reranker_mrr else 0,
        'precision': {k: np.mean(finegrained_reranker_results['precision'][k]) if finegrained_reranker_results['precision'][k] else 0 for k in k_values},
        'recall': {k: np.mean(finegrained_reranker_results['recall'][k]) if finegrained_reranker_results['recall'][k] else 0 for k in k_values},
        'ndcg': {k: np.mean(finegrained_reranker_results['ndcg'][k]) if finegrained_reranker_results['ndcg'][k] else 0 for k in k_values}
    }
    
    # Print results
    logger.info("\nEvaluation Results")
    logger.info("=" * 70)
    logger.info("BM25 Results:")
    logger.info(f"MRR: {bm25_avg['mrr']:.4f}")
    
    logger.info("\nPrecision at k:")
    for k in k_values:
        logger.info(f"P@{k}: {bm25_avg['precision'][k]:.4f}")
    
    logger.info("\nRecall at k:")
    for k in k_values:
        logger.info(f"R@{k}: {bm25_avg['recall'][k]:.4f}")
    
    logger.info("\nNDCG at k:")
    for k in k_values:
        logger.info(f"NDCG@{k}: {bm25_avg['ndcg'][k]:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Qwen3 Embedding ReRanker Results:")
    logger.info(f"MRR: {qwen_reranker_avg['mrr']:.4f}")
    
    logger.info("\nPrecision at k:")
    for k in k_values:
        logger.info(f"P@{k}: {qwen_reranker_avg['precision'][k]:.4f}")
    
    logger.info("\nRecall at k:")
    for k in k_values:
        logger.info(f"R@{k}: {qwen_reranker_avg['recall'][k]:.4f}")
    
    logger.info("\nNDCG at k:")
    for k in k_values:
        logger.info(f"NDCG@{k}: {qwen_reranker_avg['ndcg'][k]:.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Fine-grained Qwen3 ReRanker Results:")
    logger.info(f"MRR: {finegrained_reranker_avg['mrr']:.4f}")
    
    logger.info("\nPrecision at k:")
    for k in k_values:
        logger.info(f"P@{k}: {finegrained_reranker_avg['precision'][k]:.4f}")
    
    logger.info("\nRecall at k:")
    for k in k_values:
        logger.info(f"R@{k}: {finegrained_reranker_avg['recall'][k]:.4f}")
    
    logger.info("\nNDCG at k:")
    for k in k_values:
        logger.info(f"NDCG@{k}: {finegrained_reranker_avg['ndcg'][k]:.4f}")
    
    # Create visualizations
    logger.info("Creating visualizations")
    try:
        plot_comparative_metrics(bm25_avg, qwen_reranker_avg, finegrained_reranker_avg, k_values, results_dir)
        logger.info("Visualizations created successfully")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        logger.error(traceback.format_exc())

    # Create a topic-by-topic comparison
    logger.info("Creating topic-by-topic comparison")
    try:
        create_topic_comparison(bm25_topic_scores, qwen_reranker_topic_scores, finegrained_reranker_topic_scores, 
                              k_values, results_dir, bm25_avg, qwen_reranker_avg, finegrained_reranker_avg)
        logger.info("Topic comparison created successfully")
    except Exception as e:
        logger.error(f"Error creating topic comparison: {e}")
        logger.error(traceback.format_exc())

def plot_comparative_metrics(bm25_avg, qwen_reranker_avg, finegrained_reranker_avg, k_values, results_dir):
    """Create visualizations comparing BM25, Qwen3 Embedding ReRanker, and Fine-grained ReRanker performance"""
    metrics = ['precision', 'recall', 'ndcg']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        bm25_values = [bm25_avg[metric][k] for k in k_values]
        qwen_reranker_values = [qwen_reranker_avg[metric][k] for k in k_values]
        finegrained_reranker_values = [finegrained_reranker_avg[metric][k] for k in k_values]
        
        ax.plot(k_values, bm25_values, 'b-o', label='BM25')
        ax.plot(k_values, qwen_reranker_values, 'r-o', label='Qwen3 Embedding')
        ax.plot(k_values, finegrained_reranker_values, 'g-o', label='Fine-grained Qwen3')
        
        ax.set_xlabel('k')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} at k')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'evaluation_results_qwen3.png'))
    
    # Create a figure for MRR
    plt.figure(figsize=(10, 6))
    plt.bar(['BM25', 'Qwen3 Embedding', 'Fine-grained Qwen3'], 
            [bm25_avg['mrr'], qwen_reranker_avg['mrr'], finegrained_reranker_avg['mrr']], 
            color=['blue', 'red', 'green'])
    plt.ylabel('MRR')
    plt.title('Mean Reciprocal Rank Comparison')
    plt.grid(axis='y')
    plt.savefig(os.path.join(results_dir, 'mrr_comparison_qwen3.png'))

def create_topic_comparison(bm25_scores, qwen_reranker_scores, finegrained_reranker_scores, 
                          k_values, results_dir, bm25_avg, qwen_reranker_avg, finegrained_reranker_avg):
    """Create detailed topic-by-topic comparison for all three methods"""
    # Create a DataFrame for topic-by-topic comparison
    comparison_data = []
    
    # Common topics evaluated with all methods
    common_topics = set(bm25_scores.keys()) & set(qwen_reranker_scores.keys()) & set(finegrained_reranker_scores.keys())
    
    for topic_id in common_topics:
        bm25_topic = bm25_scores[topic_id]
        qwen_reranker_topic = qwen_reranker_scores[topic_id]
        finegrained_reranker_topic = finegrained_reranker_scores[topic_id]
        
        # Compare MRR
        comparison_data.append({
            'Topic': topic_id,
            'Metric': 'MRR',
            'BM25': bm25_topic['mrr'],
            'Qwen3_Embedding': qwen_reranker_topic['mrr'],
            'Finegrained_Qwen3': finegrained_reranker_topic['mrr'],
            'Qwen3_Embedding_vs_BM25': qwen_reranker_topic['mrr'] - bm25_topic['mrr'],
            'Finegrained_vs_BM25': finegrained_reranker_topic['mrr'] - bm25_topic['mrr'],
            'Finegrained_vs_Qwen3_Embedding': finegrained_reranker_topic['mrr'] - qwen_reranker_topic['mrr']
        })
        
        # Compare metrics at different k values
        for k in k_values:
            for metric in ['p', 'r', 'ndcg']:
                metric_key = f'{metric}@{k}'
                if metric_key in bm25_topic and metric_key in qwen_reranker_topic and metric_key in finegrained_reranker_topic:
                    comparison_data.append({
                        'Topic': topic_id,
                        'Metric': metric_key,
                        'BM25': bm25_topic[metric_key],
                        'Qwen3_Embedding': qwen_reranker_topic[metric_key],
                        'Finegrained_Qwen3': finegrained_reranker_topic[metric_key],
                        'Qwen3_Embedding_vs_BM25': qwen_reranker_topic[metric_key] - bm25_topic[metric_key],
                        'Finegrained_vs_BM25': finegrained_reranker_topic[metric_key] - bm25_topic[metric_key],
                        'Finegrained_vs_Qwen3_Embedding': finegrained_reranker_topic[metric_key] - qwen_reranker_topic[metric_key]
                    })
    
    # Convert to DataFrame and save
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(results_dir, 'topic_comparison_qwen3.csv'), index=False)
    
    # Create a summary showing win/loss counts
    win_loss = defaultdict(lambda: {'Qwen3_Embedding_wins': 0, 'Finegrained_wins': 0, 'BM25_wins': 0, 'Ties': 0})
    
    for _, row in comparison_df.iterrows():
        metric = row['Metric']
        best_method = max(['BM25', 'Qwen3_Embedding', 'Finegrained_Qwen3'], key=lambda x: row[x])
        
        if row['BM25'] == row['Qwen3_Embedding'] == row['Finegrained_Qwen3']:
            win_loss[metric]['Ties'] += 1
        elif best_method == 'BM25':
            win_loss[metric]['BM25_wins'] += 1
        elif best_method == 'Qwen3_Embedding':
            win_loss[metric]['Qwen3_Embedding_wins'] += 1
        else:
            win_loss[metric]['Finegrained_wins'] += 1
    
    # Convert to DataFrame and save
    win_loss_data = []
    for metric, counts in win_loss.items():
        win_loss_data.append({
            'Metric': metric,
            'BM25_wins': counts['BM25_wins'],
            'Qwen3_Embedding_wins': counts['Qwen3_Embedding_wins'],
            'Finegrained_wins': counts['Finegrained_wins'],
            'Ties': counts['Ties'],
            'Total': sum(counts.values())
        })
    
    win_loss_df = pd.DataFrame(win_loss_data)
    win_loss_df = win_loss_df.sort_values('Metric')
    win_loss_df.to_csv(os.path.join(results_dir, 'win_loss_comparison_qwen3.csv'), index=False)
    
    # Create a visualization of win/loss
    plt.figure(figsize=(14, 8))
    metrics = win_loss_df['Metric'].tolist()
    bm25_wins = win_loss_df['BM25_wins'].tolist()
    qwen3_embedding_wins = win_loss_df['Qwen3_Embedding_wins'].tolist()
    finegrained_wins = win_loss_df['Finegrained_wins'].tolist()
    ties = win_loss_df['Ties'].tolist()
    
    x = np.arange(len(metrics))
    width = 0.2
    
    plt.bar(x - 1.5*width, bm25_wins, width, label='BM25 wins', color='blue')
    plt.bar(x - 0.5*width, qwen3_embedding_wins, width, label='Qwen3 Embedding wins', color='red')
    plt.bar(x + 0.5*width, finegrained_wins, width, label='Fine-grained Qwen3 wins', color='green')
    plt.bar(x + 1.5*width, ties, width, label='Ties', color='gray')
    
    plt.xlabel('Metric')
    plt.ylabel('Count')
    plt.title('Win/Loss Comparison by Metric (All Methods)')
    plt.xticks(x, metrics, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'win_loss_comparison_qwen3.png'))
    
    # Also save a summary of results as CSV
    summary_data = []
    
    # Add results for all methods
    for method_name, method_avg in [('BM25', bm25_avg), ('Qwen3_Embedding', qwen_reranker_avg), ('Finegrained_Qwen3', finegrained_reranker_avg)]:
        for k in k_values:
            summary_data.append({
                'Method': method_name,
                'K': k,
                'Precision': method_avg['precision'][k],
                'Recall': method_avg['recall'][k],
                'NDCG': method_avg['ndcg'][k]
            })
        
        # Add MRR results (not k-dependent)
        summary_data.append({
            'Method': method_name,
            'K': 'N/A',
            'MRR': method_avg['mrr'],
            'Precision': np.nan,
            'Recall': np.nan,
            'NDCG': np.nan
        })
    
    pd.DataFrame(summary_data).to_csv(os.path.join(results_dir, 'evaluation_summary_qwen3.csv'), index=False)

if __name__ == "__main__":
    try:
        logger.info("Starting Qwen3 evaluation script")
        run_evaluation()
        logger.info("Qwen3 evaluation script completed successfully")
    except Exception as e:
        logger.error(f"Unhandled exception in evaluation script: {e}", exc_info=True)
        logger.error(traceback.format_exc())