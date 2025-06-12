import time
import faiss
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
from typing import List, Tuple, Dict, Optional
import gc

def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def print_gpu_memory(prefix=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
        print(f"{prefix}GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Free: {free:.2f}GB")

def inspect_document_stats(documents):
    """Inspect document statistics"""
    lengths = [len(doc) for doc in documents]
    print(f"Document batch info:")
    print(f"  - Number of documents: {len(documents)}")
    print(f"  - Length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
    print(f"  - Long documents (>2000 chars): {sum(1 for l in lengths if l > 2000)}")
    return lengths

class ModelManager:
    """Singleton model manager to load models once and reuse them"""
    _instance = None
    _models = {}
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._device = get_device()
            print_gpu_memory("Initial ")
        return cls._instance
    
    def get_qwen_embedding_model(self, model_name):
        """Get or load Qwen embedding model"""
        if model_name not in self._models:
            print(f"Loading Qwen embedding model {model_name} on {self._device}")
            print_gpu_memory("Before loading embedding model: ")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            
            if self._device.type == "cuda":
                try:
                    model = AutoModel.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2"
                    ).to(self._device)
                    print("Using flash_attention_2 for embedding model")
                except:
                    print("Flash attention not available, using default attention")
                    model = AutoModel.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float16
                    ).to(self._device)
            else:
                model = AutoModel.from_pretrained(model_name).to(self._device)
            
            self._models[model_name] = {
                'tokenizer': tokenizer,
                'model': model,
                'type': 'embedding'
            }
            
            print_gpu_memory("After loading embedding model: ")
            
        return self._models[model_name]
    
    def get_qwen_reranker_model(self, model_name):
        """Get or load Qwen reranker model"""
        if model_name not in self._models:
            # Clear any existing models to free memory for reranker
            self._clear_other_models(model_name)
            
            print(f"Loading Qwen reranker model {model_name} on {self._device}")
            print_gpu_memory("Before loading reranker model: ")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            
            if self._device.type == "cuda":
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2"
                    ).to(self._device).eval()
                    print("Using flash_attention_2 for reranker")
                except:
                    print("Flash attention not available, using default attention")
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16
                    ).to(self._device).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name).to(self._device).eval()
            
            token_false_id = tokenizer.convert_tokens_to_ids("no")
            token_true_id = tokenizer.convert_tokens_to_ids("yes")
            
            self._models[model_name] = {
                'tokenizer': tokenizer,
                'model': model,
                'token_false_id': token_false_id,
                'token_true_id': token_true_id,
                'type': 'reranker'
            }
            
            print_gpu_memory("After loading reranker model: ")
            
        return self._models[model_name]
    
    def _clear_other_models(self, keep_model):
        """Clear other models to free memory, keeping only the specified model"""
        for model_name in list(self._models.keys()):
            if model_name != keep_model:
                print(f"Clearing model {model_name} to free memory")
                del self._models[model_name]
                clear_gpu_memory()
    
    def get_sentence_transformer_model(self, model_name):
        """Get or load sentence transformer model"""
        if model_name not in self._models:
            print(f"Loading sentence transformer model {model_name}")
            model = SentenceTransformer(model_name)
            self._models[model_name] = {
                'model': model,
                'type': 'sentence_transformer'
            }
        return self._models[model_name]['model']

# Global model manager instance
model_manager = ModelManager()

class ReRanker:
    def __init__(self, index):
        self.index = index
        self.model_registry = []

    @staticmethod
    def initialize(model):
        embeddings = model.encode(["Place Holder"], convert_to_numpy=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        return index

    @staticmethod
    def generate_embeddings(model, text_list):
        embeddings = model.encode(text_list, convert_to_numpy=True)
        return embeddings

    @staticmethod
    def store_documents(index, embeddings):
        index.add(embeddings)
        return True

    @staticmethod
    def queryk(index, embedding, k):
        distances, indices = index.search(embedding, k)
        return (distances, indices)

    @staticmethod
    def rerank(documents, query, model='all-MiniLM-L6-v2', k=10):
        """
        Rerank the documents based on the query using the index.
        :param documents: List of documents to rerank.
        :param query: The query to use for reranking.
        :param k: The number of top documents to return.
        :return: List of reranked documents.
        """
        if model.startswith('Qwen'):
            return ReRanker._rerank_with_qwen_embedding(documents, query, model, k)
        else:
            return ReRanker._rerank_with_sentence_transformers(documents, query, model, k)

    @staticmethod
    def _rerank_with_sentence_transformers(documents, query, model_name, k):
        model = model_manager.get_sentence_transformer_model(model_name)
        faiss_index = ReRanker.initialize(model)
        embeddings = ReRanker.generate_embeddings(model, documents)
        start_time = time.time()
        ReRanker.store_documents(faiss_index, embeddings)
        index_creation_duration = time.time() - start_time

        query_embedding = ReRanker.generate_embeddings(model, [query])
        start_time = time.time()
        distances, indices = ReRanker.queryk(faiss_index, query_embedding, k=k)
        faiss_retrieval_duration = time.time() - start_time
        return distances, indices[0], faiss_retrieval_duration, index_creation_duration

    @staticmethod
    def _rerank_with_qwen_embedding(documents, query, model_name, k):
        def last_token_pool(last_hidden_states, attention_mask):
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f'Instruct: {task_description}\nQuery: {query}'

        def tokenize_single(tokenizer, text, eod_id, max_length, device):
            tokens = tokenizer(text, padding=False, truncation=True, max_length=max_length-2)
            tokens["input_ids"].append(eod_id)
            tokens["attention_mask"].append(1)
            
            # Convert to tensors
            for key in tokens:
                tokens[key] = torch.tensor([tokens[key]]).to(device)
            return tokens

        try:
            # Get model from manager (loads once, reuses afterwards)
            model_info = model_manager.get_qwen_embedding_model(model_name)
            tokenizer = model_info['tokenizer']
            model = model_info['model']
            device = model_manager._device
            
            # Inspect documents
            inspect_document_stats(documents)
            
            task = 'Given a web search query, retrieve relevant passages that answer the query'
            query_with_instruct = get_detailed_instruct(task, query)
            
            eod_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            max_length = 4096  # Reduced from 8192
            
            print(f"\nProcessing Qwen embedding for {len(documents)} documents:")
            print_gpu_memory("Before Qwen embedding processing: ")
            
            start_time = time.time()
            
            # Process query first
            print("  Processing query...")
            query_tokens = tokenize_single(tokenizer, query_with_instruct, eod_id, max_length, device)
            
            with torch.no_grad():
                query_outputs = model(**query_tokens)
                query_embedding = last_token_pool(query_outputs.last_hidden_state, query_tokens['attention_mask'])
                query_embedding = F.normalize(query_embedding, p=2, dim=1)
            
            # Clear query processing
            del query_tokens, query_outputs
            clear_gpu_memory()
            
            # Process documents one by one
            doc_embeddings = []
            successful_docs = 0
            failed_docs = 0
            
            print(f"  Processing {len(documents)} documents individually...")
            
            for i, doc in enumerate(documents):
                if i % 50 == 0:  # Progress indicator
                    print(f"    Progress: {i}/{len(documents)} documents")
                    print_gpu_memory(f"      At document {i}: ")
                
                try:
                    doc_tokens = tokenize_single(tokenizer, doc, eod_id, max_length, device)
                    
                    with torch.no_grad():
                        doc_outputs = model(**doc_tokens)
                        doc_embedding = last_token_pool(doc_outputs.last_hidden_state, doc_tokens['attention_mask'])
                        doc_embedding = F.normalize(doc_embedding, p=2, dim=1)
                        doc_embeddings.append(doc_embedding.cpu())  # Move to CPU immediately
                    
                    successful_docs += 1
                    
                    # Clear document processing
                    del doc_tokens, doc_outputs, doc_embedding
                    clear_gpu_memory()
                    
                except torch.cuda.OutOfMemoryError as e:
                    print(f"      OOM for document {i+1}: {e}")
                    # Use zero embedding as fallback
                    doc_embeddings.append(torch.zeros(1, query_embedding.shape[1]))
                    failed_docs += 1
                    clear_gpu_memory()
                    
                except Exception as e:
                    print(f"      Error for document {i+1}: {e}")
                    doc_embeddings.append(torch.zeros(1, query_embedding.shape[1]))
                    failed_docs += 1
            
            print(f"  Qwen embedding complete: {successful_docs} successful, {failed_docs} failed")
            
            # Combine all document embeddings
            if doc_embeddings:
                doc_embeddings_tensor = torch.cat(doc_embeddings, dim=0).to(device)
                
                # Compute similarities
                scores = (query_embedding @ doc_embeddings_tensor.T)[0]
                
                sorted_indices = torch.argsort(scores, descending=True)[:k]
                distances = -scores[sorted_indices].cpu().numpy()
                indices = sorted_indices.cpu().numpy()
                
                # Clean up
                del doc_embeddings_tensor, scores
                clear_gpu_memory()
            else:
                # Fallback if no embeddings generated
                distances = np.zeros(min(k, len(documents)))
                indices = np.arange(min(k, len(documents)))
            
            index_creation_duration = 0.0
            faiss_retrieval_duration = time.time() - start_time
            
            print(f"  Total Qwen embedding time: {faiss_retrieval_duration:.3f}s")
            print_gpu_memory("After Qwen embedding: ")
            
            return distances, indices, faiss_retrieval_duration, index_creation_duration
            
        except Exception as e:
            print(f"Error with Qwen model {model_name}: {e}")
            print("Falling back to sentence-transformers")
            return ReRanker._rerank_with_sentence_transformers(documents, query, 'all-MiniLM-L6-v2', k)


class FineGrainedReRanker:
    """
    Fine-grained reranker with token-level interactions using Qwen3 models.
    Processes ONE document at a time to avoid OOM issues.
    """
    
    def __init__(self, model_name='Qwen/Qwen3-Reranker-0.6B'):
        self.model_name = model_name
        
    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_single_input(self, text_pair, tokenizer, device):
        """Process a single document input with detailed inspection"""
        max_length = 3072  # Further reduced to save memory
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        
        # Tokenize single input
        inputs = tokenizer(
            [text_pair], padding=False, truncation='longest_first',
            return_attention_mask=True, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        
        # Add prefix and suffix
        inputs['input_ids'][0] = prefix_tokens + inputs['input_ids'][0] + suffix_tokens
        inputs['attention_mask'][0] = [1] * len(inputs['input_ids'][0])
        
        # Convert to tensors and pad
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        
        # Inspect input
        input_length = len(inputs['input_ids'][0])
        actual_tokens = torch.sum(inputs['attention_mask'][0]).item()
        text_length = len(text_pair)
        
        print(f"    Input inspection:")
        print(f"      - Text length: {text_length} chars")
        print(f"      - Token length: {actual_tokens} tokens (max: {max_length})")
        print(f"      - Padded length: {input_length} tokens")
        
        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(device)
            
        return inputs, actual_tokens

    @torch.no_grad()
    def compute_single_score_with_analysis(self, inputs, model, token_true_id, token_false_id, doc_idx):
        """
        Compute relevance score for a single document with detailed memory tracking.
        """
        print(f"    Processing document {doc_idx + 1}")
        print_gpu_memory("      Before model inference: ")
        
        try:
            # Forward pass with attention and hidden states
            batch_scores = model(**inputs, output_attentions=True, output_hidden_states=True)
            
            print_gpu_memory("      After model forward: ")
            
            # Extract relevance scores
            logits = batch_scores.logits[:, -1, :]
            true_vector = logits[:, token_true_id]
            false_vector = logits[:, token_false_id]
            batch_logits = torch.stack([false_vector, true_vector], dim=1)
            batch_probs = torch.nn.functional.log_softmax(batch_logits, dim=1)
            score = batch_probs[:, 1].exp().cpu().item()
            
            # Extract token-level information (move to CPU immediately)
            attentions = batch_scores.attentions[-1].cpu()  # Last layer attention [1, num_heads, seq_len, seq_len]
            hidden_states = batch_scores.hidden_states[-1].cpu()  # Last layer hidden states [1, seq_len, hidden_dim]
            
            # Compute token importance scores properly
            # attentions shape: [batch_size=1, num_heads, seq_len, seq_len]
            # We want to compute how much attention each token receives on average
            seq_len = attentions.shape[-1]
            
            # Average across heads and sum attention received by each token
            # Mean across heads: [1, seq_len, seq_len], then sum across the "from" dimension
            token_importance = attentions.mean(dim=1).sum(dim=1).squeeze()  # [seq_len]
            
            # Ensure token_importance is 1D
            if token_importance.dim() == 0:
                token_importance = token_importance.unsqueeze(0)
            
            print(f"      Score: {score:.4f}")
            print(f"      Token importance shape: {token_importance.shape}")
            print(f"      Hidden states shape: {hidden_states.shape}")
            
            # Clear intermediate results immediately
            del batch_scores, logits, true_vector, false_vector, batch_logits, batch_probs
            clear_gpu_memory()
            
            print_gpu_memory("      After cleanup: ")
            
            return {
                'score': score,
                'token_importance': token_importance,  # [seq_len]
                'token_embeddings': hidden_states.squeeze(0),  # [seq_len, hidden_dim]
                'input_ids': inputs['input_ids'].cpu().squeeze(0)  # [seq_len]
            }
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"      OOM error for document {doc_idx + 1}: {e}")
            clear_gpu_memory()
            print_gpu_memory("      After OOM cleanup: ")
            return {
                'score': 0.0,
                'token_importance': torch.tensor([]),
                'token_embeddings': torch.tensor([]),
                'input_ids': torch.tensor([])
            }
        except Exception as e:
            print(f"      Error processing document {doc_idx + 1}: {e}")
            print(f"      Error details: {type(e).__name__}: {str(e)}")
            return {
                'score': 0.0,
                'token_importance': torch.tensor([]),
                'token_embeddings': torch.tensor([]),
                'input_ids': torch.tensor([])
            }

    def extract_passage_candidates(self, analysis_results, top_k_tokens=50):
        """
        Extract top-k most important token spans as passage candidates.
        """
        passage_candidates = []
        
        for doc_idx, result in enumerate(analysis_results):
            token_importance = result['token_importance']
            input_ids = result['input_ids']
            
            if len(token_importance) == 0 or len(input_ids) == 0:
                passage_candidates.append([])
                continue
            
            try:
                # Ensure proper tensor dimensions
                importance_scores = token_importance
                tokens = input_ids
                
                # Handle different tensor shapes from single document processing
                if importance_scores.dim() > 1:
                    importance_scores = importance_scores.squeeze()
                if tokens.dim() > 1:
                    tokens = tokens.squeeze()
                
                # Debug tensor shapes
                print(f"    Doc {doc_idx}: importance_scores shape: {importance_scores.shape}, tokens shape: {tokens.shape}")
                
                if len(importance_scores) > 0 and importance_scores.numel() > 0:
                    # Ensure importance_scores is 1D
                    if importance_scores.dim() > 1:
                        importance_scores = importance_scores.flatten()
                    
                    # Get top-k tokens
                    k = min(top_k_tokens, len(importance_scores))
                    top_values, top_indices = torch.topk(importance_scores, k)
                    top_indices, _ = torch.sort(top_indices)
                    
                    # Debug top_indices
                    print(f"    Doc {doc_idx}: top_indices shape: {top_indices.shape}, first few: {top_indices[:5] if len(top_indices) > 0 else 'empty'}")
                    
                    # Convert to spans (consecutive tokens)
                    spans = []
                    if len(top_indices) > 0:
                        # Convert tensors to python ints safely
                        indices_list = [idx.item() for idx in top_indices]
                        
                        if len(indices_list) > 0:
                            current_span = [indices_list[0]]
                            
                            for i in range(1, len(indices_list)):
                                if indices_list[i] - indices_list[i-1] == 1:
                                    current_span.append(indices_list[i])
                                else:
                                    if len(current_span) >= 3:  # Minimum span length
                                        spans.append(current_span)
                                    current_span = [indices_list[i]]
                            
                            if len(current_span) >= 3:
                                spans.append(current_span)
                    
                    passage_candidates.append(spans[:5])  # Top 5 spans
                    print(f"    Doc {doc_idx}: Found {len(spans)} spans")
                else:
                    passage_candidates.append([])
                    print(f"    Doc {doc_idx}: No importance scores available")
                    
            except Exception as e:
                print(f"    Doc {doc_idx}: Error extracting passages: {e}")
                print(f"    Doc {doc_idx}: token_importance type: {type(token_importance)}, shape: {token_importance.shape if hasattr(token_importance, 'shape') else 'no shape'}")
                print(f"    Doc {doc_idx}: input_ids type: {type(input_ids)}, shape: {input_ids.shape if hasattr(input_ids, 'shape') else 'no shape'}")
                passage_candidates.append([])
        
        return passage_candidates

    def rerank_with_analysis(self, documents: List[str], query: str, k: int = 10, 
                           instruction: str = None) -> Tuple[List[float], List[int], Dict]:
        """
        Enhanced reranking with token-level analysis, processing ONE document at a time.
        """
        # Inspect input documents
        inspect_document_stats(documents)
        
        # Get model from manager
        model_info = model_manager.get_qwen_reranker_model(self.model_name)
        tokenizer = model_info['tokenizer']
        model = model_info['model']
        token_true_id = model_info['token_true_id']
        token_false_id = model_info['token_false_id']
        device = model_manager._device
        
        print(f"\nProcessing {len(documents)} documents ONE BY ONE:")
        print_gpu_memory("Starting fine-grained reranking: ")
        
        start_time = time.time()
        
        # Process each document individually
        all_results = []
        successful_docs = 0
        failed_docs = 0
        
        for i, doc in enumerate(documents):
            print(f"\n  Document {i + 1}/{len(documents)}:")
            
            try:
                # Prepare single input
                text_pair = self.format_instruction(instruction, query, doc)
                inputs, token_count = self.process_single_input(text_pair, tokenizer, device)
                
                # Process single document
                result = self.compute_single_score_with_analysis(inputs, model, token_true_id, token_false_id, i)
                all_results.append(result)
                
                if result['score'] > 0:
                    successful_docs += 1
                else:
                    failed_docs += 1
                    
                # Clear inputs from GPU
                del inputs
                clear_gpu_memory()
                
            except Exception as e:
                print(f"    Failed to process document {i + 1}: {e}")
                all_results.append({
                    'score': 0.0,
                    'token_importance': torch.tensor([]),
                    'token_embeddings': torch.tensor([]),
                    'input_ids': torch.tensor([])
                })
                failed_docs += 1
                clear_gpu_memory()
        
        print(f"\nProcessing complete: {successful_docs} successful, {failed_docs} failed")
        
        # Extract scores and sort
        scores = [result['score'] for result in all_results]
        scored_docs = [(score, idx) for idx, score in enumerate(scores)]
        scored_docs.sort(reverse=True)
        
        top_k_scores = [score for score, _ in scored_docs[:k]]
        top_k_indices = [idx for _, idx in scored_docs[:k]]
        
        # Extract passage candidates
        passage_candidates = self.extract_passage_candidates(all_results)
        
        retrieval_duration = time.time() - start_time
        
        print(f"\nFine-grained reranking completed in {retrieval_duration:.3f}s")
        print(f"Average time per document: {retrieval_duration/len(documents):.3f}s")
        print_gpu_memory("Final memory state: ")
        
        # Prepare analysis metadata
        analysis_metadata = {
            'passage_candidates': passage_candidates,
            'token_embeddings': [r['token_embeddings'] for r in all_results],
            'retrieval_duration': retrieval_duration,
            'successful_docs': successful_docs,
            'failed_docs': failed_docs,
            'model_name': self.model_name,
            'device': str(device)
        }
        
        return top_k_scores, top_k_indices, analysis_metadata

    _last_analysis_metadata = None
    
    @staticmethod
    def rerank(documents: List[str], query: str, model: str = 'Qwen/Qwen3-Reranker-0.6B', k: int = 10):
        """
        Static method interface compatible with existing ReRanker API.
        Returns format: (distances, indices, duration, index_creation_duration)
        """
        reranker = FineGrainedReRanker(model)
        scores, indices, metadata = reranker.rerank_with_analysis(documents, query, k)
        
        # Store metadata for passage retrieval access
        FineGrainedReRanker._last_analysis_metadata = metadata
        
        # Convert scores to distances (negative scores for compatibility)
        distances = [-score for score in scores]
        
        return distances, indices, metadata['retrieval_duration'], 0.0