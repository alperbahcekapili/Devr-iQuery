import os
import shutil
import numpy as np
import pandas as pd
import pyterrier as pt
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm # Ensure tqdm is imported if you use it directly; otherwise, logger is used.
import sys
import logging
import re
import traceback
import random

# Create results directory if it doesn't exist
results_dir = "./results5/"
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
    from vector.indexer import ReRanker # Original ReRanker
    from vector.fine_grained_indexer import FineGrainedReRanker # New FineGrainedReRanker
    from vector.bm25 import create_index
    logger.info("Successfully imported all modules")
except Exception as e:
    logger.error(f"Error importing modules: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Initialize PyTerrier if not already initialized
logger.info("Initializing PyTerrier")
try:
    if not pt.java.started():
        pt.java.init()
    logger.info("PyTerrier initialized successfully")
except Exception as e:
    logger.error(f"Error initializing PyTerrier: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def sanitize_query_for_terrier(query):
    logger.debug(f"Original query: '{query}'")
    query = re.sub(r'\s+', ' ', query).strip()
    # Escape or remove problematic characters for Terrier
    problematic_chars = ['?', '(', ')', '"', ':', '-', '/', '\'', '.', ',', '!', '&', '|']
    for char in problematic_chars:
        if char == '&':
            query = query.replace(char, ' and ')
        elif char == '|':
            query = query.replace(char, ' or ')
        else:
            query = query.replace(char, ' ')
    
    words = []
    for word in query.split():
        if len(word) > 30:
            word = word[:30]
        words.append(word)
    query = ' '.join(words)
    query = re.sub(r'[^\w\s]', ' ', query) # Keep only alphanumeric and spaces
    query = re.sub(r'\s+', ' ', query).strip()
    if not query or len(query) < 2: # Avoid empty or too short queries
        query = "document" # Default fallback query
    logger.debug(f"Sanitized query for Terrier: '{query}'")
    return query

def calculate_precision_at_k(retrieved_docs, relevant_docs, k):
    if not retrieved_docs or len(retrieved_docs) == 0: return 0.0
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_at_k if doc in relevant_docs]
    return len(relevant_retrieved) / min(k, len(retrieved_docs)) # Denominator is min for cases where less than k docs are retrieved

def calculate_recall_at_k(retrieved_docs, relevant_docs, k):
    if not relevant_docs or len(relevant_docs) == 0: return 1.0 
    if not retrieved_docs or len(retrieved_docs) == 0: return 0.0
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = [doc for doc in retrieved_at_k if doc in relevant_docs]
    return len(relevant_retrieved) / len(relevant_docs)

def calculate_mrr(retrieved_docs, relevant_docs):
    if not retrieved_docs or not relevant_docs: return 0.0
    for i, doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0

def calculate_dcg_at_k(retrieved_docs, relevance_scores_map, k): # Changed relevance_scores to map for clarity
    if not retrieved_docs: return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        rel = float(relevance_scores_map.get(doc_id, 0))
        dcg += (2 ** rel - 1) / np.log2(i + 2) # i+2 because ranks are 1-based, log starts from 2 for denom
    return dcg

def calculate_ndcg_at_k(retrieved_docs, relevance_scores_map, k):
    if not retrieved_docs: return 0.0
    dcg = calculate_dcg_at_k(retrieved_docs, relevance_scores_map, k)
    
    # Ideal DCG: Sort all known relevant documents by their true relevance scores
    # Consider only documents present in relevance_scores_map for IDCG calculation
    ideal_sorted_docs = sorted(
        [doc_id for doc_id in relevance_scores_map.keys()], 
        key=lambda doc_id: relevance_scores_map.get(doc_id, 0), 
        reverse=True
    )
    idcg = calculate_dcg_at_k(ideal_sorted_docs, relevance_scores_map, k)
    
    if idcg == 0: return 0.0
    return dcg / idcg

def safely_remove_directory(directory_path):
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
        logger.info(f"Loaded {len(documents)} documents, {len(topics)} topics, and qrels for {len(qrels)} topics.")
        # ... (data integrity checks from original script) ...
    except Exception as e:
        logger.error(f"Error loading data: {e}\n{traceback.format_exc()}")
        return

    docs_for_indexing = [{"docno": doc_id, "text": doc.get("headline", "") + " " + doc.get("text", "")}
                         for doc_id, doc in documents.items()]
    docs_df = pd.DataFrame(docs_for_indexing)
    logger.info(f"Prepared {len(docs_df)} documents for indexing.")
    max_text_length = docs_df['text'].str.len().max() if not docs_df.empty else 2048
    logger.info(f"Maximum text length in documents: {max_text_length}")

    index_path = os.path.join(results_dir, "index")
    if not safely_remove_directory(index_path): # Attempt to remove for a clean run
        logger.warning("Could not remove existing index directory. Attempting to proceed.")
    
    try:
        logger.info(f"Creating new index at {index_path}")
        # Adjust meta length for text based on observed max_text_length
        indexer = pt.IterDictIndexer(index_path, meta={'docno': 30, 'text': int(max_text_length * 1.1) + 100},overwrite=True)
        indexref = indexer.index(docs_df.to_dict(orient="records"))
        index = pt.IndexFactory.of(indexref)
        logger.info(f"Index created successfully with {index.getCollectionStatistics().getNumberOfDocuments()} documents")
        bm25 = pt.BatchRetrieve(index, wmodel="BM25", controls={"c": 0.75, "bm25.b": 0.75, "bm25.k_1": 1.2})
        logger.info("BM25 retriever initialized")
    except Exception as e:
        logger.error(f"Error creating/loading index: {e}\n{traceback.format_exc()}")
        return

    # Initialize FineGrainedReRanker
    logger.info("Initializing FineGrainedReRanker")
    fg_reranker = None
    try:
        fg_reranker = FineGrainedReRanker() 
        logger.info("FineGrainedReRanker initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing FineGrainedReRanker: {e}\n{traceback.format_exc()}")
        logger.warning("Proceeding without FineGrainedReRanker.")

    k_values = [5, 10, 20, 50, 100]
    metrics_to_collect = ['precision', 'recall', 'ndcg']
    
    bm25_results = {metric: {k: [] for k in k_values} for metric in metrics_to_collect}
    reranker_results = {metric: {k: [] for k in k_values} for metric in metrics_to_collect}
    fg_reranker_results = {metric: {k: [] for k in k_values} for metric in metrics_to_collect}
    
    bm25_mrr, reranker_mrr, fg_reranker_mrr = [], [], []
    
    bm25_topic_scores, reranker_topic_scores, fg_reranker_topic_scores = {}, {}, {}
    
    # ... (logging setup and other initializations from original script) ...
    topics_evaluated, topics_skipped = 0, 0
    bm25_failures, reranker_failures, fg_reranker_failures = 0, 0, 0

    docid_to_text = {doc_id: doc.get("headline", "") + " " + doc.get("text", "") 
                     for doc_id, doc in documents.items()}
    
    doc_ids_in_collection = set(documents.keys())

    valid_topics_for_eval = []
    for topic_id, topic_data in topics.items():
        if topic_id not in qrels: continue
        relevant_doc_ids_for_topic = [doc_id for doc_id, rel_score in qrels[topic_id].items() if int(rel_score) > 0]
        relevant_in_collection_count = len([doc_id for doc_id in relevant_doc_ids_for_topic if doc_id in doc_ids_in_collection])
        if relevant_in_collection_count > 0:
            valid_topics_for_eval.append((topic_id, topic_data, relevant_in_collection_count, len(relevant_doc_ids_for_topic)))
    
    logger.info(f"Found {len(valid_topics_for_eval)} topics with relevant documents in our collection.")
    valid_topics_for_eval.sort(key=lambda x: x[2], reverse=True) # Sort by available relevant docs

    for topic_id, topic_content, _, _ in tqdm(valid_topics_for_eval, desc="Evaluating Topics"):
        logger.info(f"\nProcessing topic {topic_id}: {topic_content.get('title', 'N/A')}")
        
        query_title = topic_content.get("title", "").strip()
        query_desc = topic_content.get("description", "").strip()
        base_query = f"{query_title} {query_desc}".strip()
        
        terrier_query = sanitize_query_for_terrier(base_query)
        reranker_general_query = re.sub(r'\s+', ' ', base_query).strip()
        
        current_qrels = qrels[topic_id]
        # Relevant docs for this topic that are IN OUR DOCUMENT COLLECTION
        true_relevant_docs_in_collection = [doc_id for doc_id, rel_score in current_qrels.items() 
                                            if int(rel_score) > 0 and doc_id in doc_ids_in_collection]
        # Relevance map for NDCG, containing only docs in our collection
        relevance_map_for_ndcg = {doc_id: int(rel_score) for doc_id, rel_score in current_qrels.items()
                                  if doc_id in doc_ids_in_collection and int(rel_score) > 0}


        if not true_relevant_docs_in_collection:
            logger.warning(f"Skipping topic {topic_id}: No relevant documents found in our loaded collection.")
            topics_skipped +=1
            continue

        # BM25 Retrieval
        bm25_retrieved_ids = []
        try:
            query_df = pd.DataFrame([{"qid": topic_id, "query": terrier_query}])
            bm25_df = bm25.transform(query_df)
            if not bm25_df.empty and 'docno' in bm25_df.columns:
                bm25_retrieved_ids = bm25_df["docno"].tolist()
            logger.info(f"BM25 retrieved {len(bm25_retrieved_ids)} docs for topic {topic_id}")
        except Exception as e:
            logger.error(f"BM25 failed for topic {topic_id}: {e}\n{traceback.format_exc()}")
            bm25_failures +=1

        # Prepare candidates for rerankers
        candidate_doc_ids = set(true_relevant_docs_in_collection) # Start with all known relevant
        candidate_doc_ids.update(bm25_retrieved_ids[:300]) # Add top BM25
        
        # If still too few, add random docs (ensure they are in docid_to_text)
        if len(candidate_doc_ids) < 50 and len(docid_to_text) > len(candidate_doc_ids) : # Ensure there are docs to sample
            sample_size = min(100, len(docid_to_text) - len(candidate_doc_ids))
            available_for_sample = [docid for docid in docid_to_text if docid not in candidate_doc_ids]
            if available_for_sample and sample_size > 0 :
                 candidate_doc_ids.update(random.sample(available_for_sample, min(sample_size, len(available_for_sample))))

        reranker_candidate_texts = [docid_to_text[doc_id] for doc_id in candidate_doc_ids if doc_id in docid_to_text]
        reranker_candidate_docids_map = [doc_id for doc_id in candidate_doc_ids if doc_id in docid_to_text]


        # Original ReRanker
        orig_reranker_retrieved_ids = []
        try:
            if reranker_candidate_texts:
                # ReRanker.rerank returns (distances, indices)
                _, indices = ReRanker.rerank(reranker_candidate_texts, reranker_general_query, k=min(len(reranker_candidate_texts), max(k_values)))
                orig_reranker_retrieved_ids = [reranker_candidate_docids_map[idx] for idx in indices if idx < len(reranker_candidate_docids_map)]
                logger.info(f"Original ReRanker retrieved {len(orig_reranker_retrieved_ids)} docs for topic {topic_id}")
            else: logger.warning(f"No candidates for Original ReRanker on topic {topic_id}")
        except Exception as e:
            logger.error(f"Original ReRanker failed for topic {topic_id}: {e}\n{traceback.format_exc()}")
            reranker_failures += 1
            
        # FineGrainedReRanker
        fg_reranker_retrieved_ids = []
        if fg_reranker: # Check if initialized
            try:
                if reranker_candidate_texts:
                     # FineGrainedReRanker.rerank returns (scores, indices)
                    _, indices = fg_reranker.rerank(reranker_candidate_texts, reranker_general_query, k=min(len(reranker_candidate_texts), max(k_values)))
                    fg_reranker_retrieved_ids = [reranker_candidate_docids_map[idx] for idx in indices if idx < len(reranker_candidate_docids_map)]
                    logger.info(f"FineGrainedReRanker retrieved {len(fg_reranker_retrieved_ids)} docs for topic {topic_id}")
                else: logger.warning(f"No candidates for FineGrainedReRanker on topic {topic_id}")
            except Exception as e:
                logger.error(f"FineGrainedReRanker failed for topic {topic_id}: {e}\n{traceback.format_exc()}")
                fg_reranker_failures +=1
        
        # Calculate and store metrics
        systems_results = {
            "BM25": (bm25_retrieved_ids, bm25_results, bm25_mrr, bm25_topic_scores),
            "ReRanker (Original)": (orig_reranker_retrieved_ids, reranker_results, reranker_mrr, reranker_topic_scores),
            "FineGrainedReRanker": (fg_reranker_retrieved_ids, fg_reranker_results, fg_reranker_mrr, fg_reranker_topic_scores)
        }

        for sys_name, (retrieved_list, sys_metrics_dict, sys_mrr_list, sys_topic_scores_dict) in systems_results.items():
            if not retrieved_list and sys_name != "FineGrainedReRanker" and sys_name !="ReRanker (Original)": # BM25 must have results to proceed for rerankers usually
                 if sys_name == "BM25": logger.warning(f"No results for {sys_name} on topic {topic_id}, skipping its metrics.")
                 # If FineGrainedReRanker is None, it won't be in systems_results for metrics
                 if sys_name == "FineGrainedReRanker" and not fg_reranker: continue
                 # if retrieved_list is empty for rerankers, it means they failed or had no input
                 if not retrieved_list :
                     logger.warning(f"No results for {sys_name} on topic {topic_id}, skipping its metrics this topic.")
                     # Add 0 for all metrics for this topic if it failed to produce results
                     mrr_val = 0.0
                     sys_mrr_list.append(mrr_val)
                     topic_metric_vals = {'mrr': mrr_val}
                     for k_val in k_values:
                        for metric_name in metrics_to_collect:
                            sys_metrics_dict[metric_name][k_val].append(0.0)
                            topic_metric_vals[f'{metric_name[0]}@{k_val}'] = 0.0 # p@k, r@k, n@k
                     sys_topic_scores_dict[topic_id] = topic_metric_vals
                     continue


            mrr_val = calculate_mrr(retrieved_list, true_relevant_docs_in_collection)
            sys_mrr_list.append(mrr_val)
            logger.info(f"{sys_name} MRR for topic {topic_id}: {mrr_val:.4f}")
            topic_metric_vals = {'mrr': mrr_val}

            for k_val in k_values:
                p_at_k = calculate_precision_at_k(retrieved_list, true_relevant_docs_in_collection, k_val)
                r_at_k = calculate_recall_at_k(retrieved_list, true_relevant_docs_in_collection, k_val)
                n_at_k = calculate_ndcg_at_k(retrieved_list, relevance_map_for_ndcg, k_val)
                
                sys_metrics_dict['precision'][k_val].append(p_at_k)
                sys_metrics_dict['recall'][k_val].append(r_at_k)
                sys_metrics_dict['ndcg'][k_val].append(n_at_k)
                
                topic_metric_vals[f'p@{k_val}'] = p_at_k
                topic_metric_vals[f'r@{k_val}'] = r_at_k
                topic_metric_vals[f'ndcg@{k_val}'] = n_at_k
                logger.info(f"{sys_name} metrics for topic {topic_id} at k={k_val}: P={p_at_k:.4f}, R={r_at_k:.4f}, NDCG={n_at_k:.4f}")
            sys_topic_scores_dict[topic_id] = topic_metric_vals
        topics_evaluated +=1

    logger.info(f"\nEvaluation complete. Evaluated {topics_evaluated} topics. Skipped {topics_skipped} topics.")
    logger.info(f"Failures: BM25={bm25_failures}, OriginalReRanker={reranker_failures}, FineGrainedReRanker={fg_reranker_failures}")

    # Calculate Averages
    all_averages = {}
    for sys_name, (_, sys_metrics_dict, sys_mrr_list, _) in systems_results.items():
        if sys_name == "FineGrainedReRanker" and not fg_reranker: continue # Skip if not initialized

        avg_metrics = {
            'mrr': np.mean(sys_mrr_list) if sys_mrr_list else 0,
            'precision': {k: np.mean(sys_metrics_dict['precision'][k]) if sys_metrics_dict['precision'][k] else 0 for k in k_values},
            'recall': {k: np.mean(sys_metrics_dict['recall'][k]) if sys_metrics_dict['recall'][k] else 0 for k in k_values},
            'ndcg': {k: np.mean(sys_metrics_dict['ndcg'][k]) if sys_metrics_dict['ndcg'][k] else 0 for k in k_values}
        }
        all_averages[sys_name] = avg_metrics
        
        logger.info(f"\n{sys_name} Average Results:")
        logger.info(f"MRR: {avg_metrics['mrr']:.4f}")
        for metric_name in metrics_to_collect:
            logger.info(f"\nAverage {metric_name.capitalize()} at k:")
            for k_val in k_values: logger.info(f"{metric_name.capitalize()}@{k_val}: {avg_metrics[metric_name][k_val]:.4f}")

    # Visualizations and Detailed Comparisons
    try:
        plot_comparative_metrics(all_averages.get("BM25"), 
                                 all_averages.get("ReRanker (Original)"), 
                                 all_averages.get("FineGrainedReRanker") if fg_reranker else None, 
                                 k_values, results_dir)
        
        create_topic_comparison(bm25_topic_scores, 
                                reranker_topic_scores, 
                                fg_reranker_topic_scores if fg_reranker else {}, 
                                k_values, results_dir, 
                                all_averages.get("BM25"),
                                all_averages.get("ReRanker (Original)"),
                                all_averages.get("FineGrainedReRanker") if fg_reranker else None
                                )
        logger.info("Visualizations and detailed comparisons created successfully.")
    except Exception as e:
        logger.error(f"Error during post-evaluation analysis: {e}\n{traceback.format_exc()}")


def plot_comparative_metrics(bm25_avg, reranker_orig_avg, fg_reranker_avg, k_values, results_dir):
    if not bm25_avg: # Should always be there if BM25 ran
        logger.warning("BM25 average results not available for plotting.")
        return

    metrics_to_plot = ['precision', 'recall', 'ndcg']
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_metrics, figsize=(8 * num_metrics, 6))
    if num_metrics == 1: axes = [axes] # Ensure axes is iterable

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        ax.plot(k_values, [bm25_avg[metric][k] for k in k_values], 'b-o', label='BM25')
        if reranker_orig_avg:
            ax.plot(k_values, [reranker_orig_avg[metric][k] for k in k_values], 'r-s', label='ReRanker (Original)')
        if fg_reranker_avg:
            ax.plot(k_values, [fg_reranker_avg[metric][k] for k in k_values], 'g-^', label='FineGrainedReRanker')
        
        ax.set_xlabel('k')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()}@k Comparison')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_at_k_comparison.png'))
    plt.close(fig)

    # MRR Plot
    plt.figure(figsize=(10, 7))
    mrr_methods, mrr_values, mrr_colors = [], [], []
    mrr_methods.append('BM25'); mrr_values.append(bm25_avg['mrr']); mrr_colors.append('blue')
    if reranker_orig_avg:
        mrr_methods.append('ReRanker (Original)'); mrr_values.append(reranker_orig_avg['mrr']); mrr_colors.append('red')
    if fg_reranker_avg:
        mrr_methods.append('FineGrainedReRanker'); mrr_values.append(fg_reranker_avg['mrr']); mrr_colors.append('green')
        
    plt.bar(mrr_methods, mrr_values, color=mrr_colors)
    plt.ylabel('Mean Reciprocal Rank (MRR)')
    plt.title('MRR Comparison')
    plt.grid(axis='y', linestyle='--')
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'mrr_comparison.png'))
    plt.close()

def generate_win_loss_analysis(comparison_df, method1_col, method2_col, results_dir, file_prefix, k_values):
    win_loss_data = []
    metrics_for_win_loss = ['MRR'] + [f'{m}@{k}' for m in ['p','r','ndcg'] for k in k_values]

    for metric_name in metrics_for_win_loss:
        # Filter DataFrame for the specific metric
        metric_df = comparison_df[comparison_df['Metric'] == metric_name]
        if metric_df.empty: continue

        wins_method2 = 0
        wins_method1 = 0
        ties = 0
        
        for _, row in metric_df.iterrows():
            val_method1 = row.get(method1_col, 0.0) # Default to 0 if a method's score is missing for a topic
            val_method2 = row.get(method2_col, 0.0)
            diff = val_method2 - val_method1
            
            if diff > 0.001: wins_method2 += 1
            elif diff < -0.001: wins_method1 += 1
            else: ties += 1
        
        win_loss_data.append({
            'Metric': metric_name,
            f'{method2_col}_wins': wins_method2,
            f'{method1_col}_wins': wins_method1,
            'Ties': ties,
            'Total_Topics': wins_method1 + wins_method2 + ties
        })

    if not win_loss_data:
        logger.warning(f"No data for win/loss analysis between {method1_col} and {method2_col}")
        return

    win_loss_df = pd.DataFrame(win_loss_data)
    win_loss_df.to_csv(os.path.join(results_dir, f'{file_prefix}_win_loss.csv'), index=False)

    # Plotting win/loss
    plt.figure(figsize=(14, 8)) # Adjusted for more metrics
    plot_metrics = win_loss_df['Metric'].tolist()
    method2_wins_plot = win_loss_df[f'{method2_col}_wins'].tolist()
    method1_wins_plot = win_loss_df[f'{method1_col}_wins'].tolist()
    ties_plot = win_loss_df['Ties'].tolist()
    
    x_indices = np.arange(len(plot_metrics))
    bar_width = 0.25
    
    plt.bar(x_indices - bar_width, method2_wins_plot, bar_width, label=f'{method2_col} Wins', color='forestgreen')
    plt.bar(x_indices, method1_wins_plot, bar_width, label=f'{method1_col} Wins', color='royalblue')
    plt.bar(x_indices + bar_width, ties_plot, bar_width, label='Ties', color='silver')
    
    plt.xlabel('Metric')
    plt.ylabel('Number of Topics')
    plt.title(f'Win/Loss/Tie Comparison: {method2_col} vs {method1_col}')
    plt.xticks(x_indices, plot_metrics, rotation=60, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{file_prefix}_win_loss_plot.png'))
    plt.close()


def create_topic_comparison(bm25_topic_scores, reranker_orig_topic_scores, fg_reranker_topic_scores, 
                            k_values, results_dir, 
                            bm25_avg, reranker_orig_avg, fg_reranker_avg):
    
    comparison_data = []
    # Ensure all topic IDs are strings for consistent merging/set operations
    all_topic_ids = set(map(str, bm25_topic_scores.keys())) | \
                    set(map(str, reranker_orig_topic_scores.keys())) | \
                    set(map(str, fg_reranker_topic_scores.keys()))

    for topic_id_str in sorted(list(all_topic_ids)):
        # Get scores for each method, defaulting to empty dict if topic not scored by a method
        bm25_s = bm25_topic_scores.get(topic_id_str, {})
        orig_r_s = reranker_orig_topic_scores.get(topic_id_str, {})
        fg_r_s = fg_reranker_topic_scores.get(topic_id_str, {})

        row_base = {'Topic': topic_id_str}
        metrics_to_log = ['mrr'] + [f'{m}@{k}' for m in ['p', 'r', 'ndcg'] for k in k_values]

        for metric_key in metrics_to_log:
            row = row_base.copy()
            row['Metric'] = metric_key
            row['BM25'] = bm25_s.get(metric_key) # .get() handles missing metrics for a topic gracefully
            row['ReRanker_Orig'] = orig_r_s.get(metric_key)
            row['FG_ReRanker'] = fg_r_s.get(metric_key)
            comparison_data.append(row)
            
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(results_dir, 'topic_by_topic_scores.csv'), index=False)
    logger.info(f"Saved detailed topic-by-topic scores to topic_by_topic_scores.csv")

    # Generate pairwise win/loss analyses if data is available
    if not comparison_df.empty:
        if 'ReRanker_Orig' in comparison_df.columns and 'BM25' in comparison_df.columns:
             generate_win_loss_analysis(comparison_df, 'BM25', 'ReRanker_Orig', results_dir, 'OrigReRanker_vs_BM25', k_values)
        if fg_reranker_avg and 'FG_ReRanker' in comparison_df.columns and 'BM25' in comparison_df.columns: # Check if FG_ReRanker was run
             generate_win_loss_analysis(comparison_df, 'BM25', 'FG_ReRanker', results_dir, 'FGReRanker_vs_BM25', k_values)
        if fg_reranker_avg and 'FG_ReRanker' in comparison_df.columns and 'ReRanker_Orig' in comparison_df.columns:
             generate_win_loss_analysis(comparison_df, 'ReRanker_Orig', 'FG_ReRanker', results_dir, 'FGReRanker_vs_OrigReRanker', k_values)


    # Summary CSV of average results
    summary_data = []
    systems_avg = {
        "BM25": bm25_avg, 
        "ReRanker (Original)": reranker_orig_avg, 
        "FineGrainedReRanker": fg_reranker_avg
    }

    for method_name, avg_scores in systems_avg.items():
        if not avg_scores: continue # Skip if a method was not run/had no results

        summary_data.append({
            'Method': method_name, 'Metric': 'MRR', 'K': 'N/A', 'Average_Score': avg_scores['mrr']
        })
        for k_val in k_values:
            for metric_name in ['precision', 'recall', 'ndcg']:
                summary_data.append({
                    'Method': method_name,
                    'Metric': metric_name.capitalize(),
                    'K': k_val,
                    'Average_Score': avg_scores[metric_name][k_val]
                })
    pd.DataFrame(summary_data).to_csv(os.path.join(results_dir, 'average_evaluation_summary.csv'), index=False)
    logger.info(f"Saved average evaluation summary to average_evaluation_summary.csv")


if __name__ == "__main__":
    try:
        logger.info("Starting evaluation script")
        run_evaluation()
        logger.info("Evaluation script completed successfully")
    except Exception as e:
        logger.error(f"Unhandled exception in evaluation script: {e}", exc_info=True)