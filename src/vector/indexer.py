import time
import faiss
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
from typing import List, Tuple, Dict, Optional

# Device detection and management
def get_device():
    """Get the best available device (CUDA > MPS > CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

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
        model = SentenceTransformer(model_name)
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

        def tokenize(tokenizer, input_texts, eod_id, max_length, device):
            batch_dict = tokenizer(input_texts, padding=False, truncation=True, max_length=max_length-2)
            for seq, att in zip(batch_dict["input_ids"], batch_dict["attention_mask"]):
                seq.append(eod_id)
                att.append(1)
            batch_dict = tokenizer.pad(batch_dict, padding=True, return_tensors="pt")
            # Move to device
            for key in batch_dict:
                batch_dict[key] = batch_dict[key].to(device)
            return batch_dict

        try:
            device = get_device()
            
            print(f"Loading Qwen model {model_name} on {device}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
            
            # Load model with appropriate settings based on device
            if device.type == "cuda":
                # Use float16 and flash attention for GPU
                try:
                    model = AutoModel.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2"
                    ).to(device)
                    print("Using flash_attention_2 for better performance")
                except:
                    print("Flash attention not available, using default attention")
                    model = AutoModel.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float16
                    ).to(device)
            else:
                # Use float32 for CPU/MPS
                model = AutoModel.from_pretrained(model_name).to(device)
            
            task = 'Given a web search query, retrieve relevant passages that answer the query'
            query_with_instruct = get_detailed_instruct(task, query)
            input_texts = [query_with_instruct] + documents
            
            eod_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            max_length = 8192
            
            start_time = time.time()
            batch_dict = tokenize(tokenizer, input_texts, eod_id, max_length, device)
            
            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            query_embedding = embeddings[:1]
            doc_embeddings = embeddings[1:]
            scores = (query_embedding @ doc_embeddings.T)[0]
            
            sorted_indices = torch.argsort(scores, descending=True)[:k]
            distances = -scores[sorted_indices].cpu().numpy()
            indices = sorted_indices.cpu().numpy()
            
            index_creation_duration = 0.0
            faiss_retrieval_duration = time.time() - start_time
            
            print(f"Qwen embedding completed in {faiss_retrieval_duration:.3f}s on {device}")
            
            return distances, indices, faiss_retrieval_duration, index_creation_duration
            
        except Exception as e:
            print(f"Error with Qwen model {model_name}: {e}")
            print("Falling back to sentence-transformers")
            return ReRanker._rerank_with_sentence_transformers(documents, query, 'all-MiniLM-L6-v2', k)


class FineGrainedReRanker:
    """
    Fine-grained reranker with token-level interactions using Qwen3 models.
    Designed for future passage retrieval and direct LLM integration.
    """
    
    def __init__(self, model_name='Qwen/Qwen3-Reranker-0.6B'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self.token_embeddings_cache = {}
        
    def _load_model(self):
        if self.model is None:
            try:
                self.device = get_device()
                
                print(f"Loading fine-grained reranker {self.model_name} on {self.device}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')
                
                # Load model with appropriate settings based on device
                if self.device.type == "cuda":
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float16,
                            attn_implementation="flash_attention_2"
                        ).to(self.device).eval()
                        print("Using flash_attention_2 for fine-grained reranker")
                    except:
                        print("Flash attention not available for reranker, using default attention")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            torch_dtype=torch.float16
                        ).to(self.device).eval()
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device).eval()
                
                self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
                self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
                
                print(f"Fine-grained reranker loaded successfully on {self.device}")
                
            except Exception as e:
                print(f"Failed to load Qwen3 reranker: {e}")
                raise

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs):
        max_length = 8192
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)
        
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
            
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        
        # Move to device
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
            
        return inputs

    @torch.no_grad()
    def compute_scores_with_token_analysis(self, inputs):
        """
        Compute relevance scores with token-level interaction analysis.
        Returns both scores and token-level attention patterns for passage retrieval.
        """
        batch_scores = self.model(**inputs, output_attentions=True, output_hidden_states=True)
        
        # Extract relevance scores
        logits = batch_scores.logits[:, -1, :]
        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]
        batch_logits = torch.stack([false_vector, true_vector], dim=1)
        batch_probs = torch.nn.functional.log_softmax(batch_logits, dim=1)
        scores = batch_probs[:, 1].exp().tolist()
        
        # Extract token-level information for future passage retrieval
        attentions = batch_scores.attentions[-1]  # Last layer attention
        hidden_states = batch_scores.hidden_states[-1]  # Last layer hidden states
        
        # Compute token importance scores (average attention weights)
        token_importance = attentions.mean(dim=1).squeeze()  # [batch_size, seq_len]
        
        # Store embeddings for potential LLM integration
        token_embeddings = hidden_states  # [batch_size, seq_len, hidden_dim]
        
        return {
            'scores': scores,
            'token_importance': token_importance,
            'token_embeddings': token_embeddings,
            'input_ids': inputs['input_ids']
        }

    def extract_passage_candidates(self, analysis_result, top_k_tokens=50):
        """
        Extract top-k most important token spans as passage candidates.
        Designed for future passage retrieval implementation.
        """
        token_importance = analysis_result['token_importance']
        input_ids = analysis_result['input_ids']
        
        passage_candidates = []
        for batch_idx in range(token_importance.shape[0]):
            importance_scores = token_importance[batch_idx]
            tokens = input_ids[batch_idx]
            
            # Find top-k important tokens
            top_indices = torch.topk(importance_scores, min(top_k_tokens, len(importance_scores)))[1]
            top_indices = top_indices.sort()[0]
            
            # Convert to spans (consecutive tokens)
            spans = []
            current_span = [top_indices[0].item()]
            
            for i in range(1, len(top_indices)):
                if top_indices[i] - top_indices[i-1] == 1:
                    current_span.append(top_indices[i].item())
                else:
                    if len(current_span) >= 3:  # Minimum span length
                        spans.append(current_span)
                    current_span = [top_indices[i].item()]
            
            if len(current_span) >= 3:
                spans.append(current_span)
            
            # Extract text for each span
            span_texts = []
            for span in spans[:5]:  # Top 5 spans
                span_tokens = tokens[span[0]:span[-1]+1]
                span_text = self.tokenizer.decode(span_tokens, skip_special_tokens=True)
                span_texts.append({
                    'text': span_text,
                    'start_idx': span[0],
                    'end_idx': span[-1],
                    'importance': importance_scores[span].mean().item()
                })
            
            passage_candidates.append(span_texts)
        
        return passage_candidates

    def rerank_with_analysis(self, documents: List[str], query: str, k: int = 10, 
                           instruction: str = None) -> Tuple[List[float], List[int], Dict]:
        """
        Enhanced reranking with token-level analysis for fine-grained understanding.
        """
        self._load_model()
        
        start_time = time.time()
        
        # Prepare input pairs
        pairs = [self.format_instruction(instruction, query, doc) for doc in documents]
        inputs = self.process_inputs(pairs)
        
        # Compute scores with token analysis
        analysis_result = self.compute_scores_with_token_analysis(inputs)
        scores = analysis_result['scores']
        
        # Extract passage candidates for future use
        passage_candidates = self.extract_passage_candidates(analysis_result)
        
        # Sort and return top-k
        scored_docs = [(score, idx) for idx, score in enumerate(scores)]
        scored_docs.sort(reverse=True)
        
        top_k_scores = [score for score, _ in scored_docs[:k]]
        top_k_indices = [idx for _, idx in scored_docs[:k]]
        
        retrieval_duration = time.time() - start_time
        
        print(f"Fine-grained reranking completed in {retrieval_duration:.3f}s on {self.device}")
        
        # Prepare analysis metadata for future passage retrieval and LLM integration
        analysis_metadata = {
            'passage_candidates': passage_candidates,
            'token_embeddings': analysis_result['token_embeddings'],
            'retrieval_duration': retrieval_duration,
            'model_name': self.model_name,
            'device': str(self.device)
        }
        
        return top_k_scores, top_k_indices, analysis_metadata

    @staticmethod
    def rerank(documents: List[str], query: str, model: str = 'Qwen/Qwen3-Reranker-0.6B', k: int = 10):
        """
        Static method interface compatible with existing ReRanker API.
        Returns format: (distances, indices, duration, index_creation_duration)
        """
        reranker = FineGrainedReRanker(model)
        scores, indices, metadata = reranker.rerank_with_analysis(documents, query, k)
        
        # Convert scores to distances (negative scores for compatibility)
        distances = [-score for score in scores]
        
        return distances, indices, metadata['retrieval_duration'], 0.0