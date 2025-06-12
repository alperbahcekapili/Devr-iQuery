import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import re

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available for PassageRetriever")

class PassageRetriever:
    """
    Passage retrieval system that uses fine-grained token analysis
    to extract and rank the most relevant passages from documents.
    """
    
    def __init__(self, tokenizer_name='Qwen/Qwen3-Reranker-0.6B'):
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:
                print(f"Warning: Could not load tokenizer {tokenizer_name}: {e}")
        
    def extract_passages_from_analysis(self, analysis_metadata: Dict, documents: List[str], 
                                     query: str, top_k_passages: int = 5) -> List[Dict]:
        """
        Extract top-k most relevant passages from documents using token-level analysis.
        
        Returns:
            List of passage dictionaries with text, score, document_id, and highlighting info
        """
        passage_candidates = analysis_metadata.get('passage_candidates', [])
        token_embeddings = analysis_metadata.get('token_embeddings', [])
        
        all_passages = []
        
        for doc_idx, doc_spans in enumerate(passage_candidates):
            if doc_idx >= len(documents):
                continue
                
            document_text = documents[doc_idx]
            
            # Get corresponding token embeddings if available
            doc_embeddings = token_embeddings[doc_idx] if doc_idx < len(token_embeddings) else torch.tensor([])
            
            # Extract passages from token spans
            doc_passages = self._extract_passages_from_spans(
                doc_spans, document_text, doc_embeddings, doc_idx, query
            )
            all_passages.extend(doc_passages)
        
        # Sort by relevance score and return top-k
        all_passages.sort(key=lambda x: x['relevance_score'], reverse=True)
        return all_passages[:top_k_passages]
    
    def _extract_passages_from_spans(self, spans: List[List[int]], document_text: str, 
                                   embeddings: torch.Tensor, doc_idx: int, query: str) -> List[Dict]:
        """
        Extract readable passages from token spans with relevance scoring.
        """
        if not spans or len(embeddings) == 0:
            # Fallback: extract sentences and score them
            return self._extract_sentence_passages(document_text, doc_idx, query)
        
        passages = []
        
        try:
            # If tokenizer is available, use token-based extraction
            if self.tokenizer is not None:
                # Tokenize the full document to map spans to text
                doc_tokens = self.tokenizer(document_text, return_offsets_mapping=True, 
                                          add_special_tokens=False, truncation=True, max_length=3072)
                
                for span_idx, span in enumerate(spans):
                    if not span or len(span) < 3:
                        continue
                        
                    try:
                        # Get token offsets for the span
                        start_offset = doc_tokens['offset_mapping'][span[0]][0] if span[0] < len(doc_tokens['offset_mapping']) else 0
                        end_offset = doc_tokens['offset_mapping'][span[-1]][1] if span[-1] < len(doc_tokens['offset_mapping']) else len(document_text)
                        
                        # Extract text passage
                        passage_text = document_text[start_offset:end_offset].strip()
                        
                        # Expand to sentence boundaries for better readability
                        expanded_passage = self._expand_to_sentence_boundaries(document_text, start_offset, end_offset)
                        
                        # Calculate relevance score based on token importance
                        if hasattr(embeddings, 'shape') and len(embeddings.shape) > 1:
                            span_embeddings = embeddings[span[0]:span[-1]+1] if span[-1] < embeddings.shape[0] else embeddings[span[0]:]
                            relevance_score = float(torch.mean(torch.norm(span_embeddings, dim=-1))) if len(span_embeddings) > 0 else 0.0
                        else:
                            relevance_score = 0.5  # Default score
                        
                        passages.append({
                            'text': expanded_passage,
                            'core_text': passage_text,
                            'relevance_score': relevance_score,
                            'document_id': doc_idx,
                            'span_id': span_idx,
                            'start_offset': start_offset,
                            'end_offset': end_offset,
                            'highlight_start': start_offset,
                            'highlight_end': end_offset,
                            'query': query
                        })
                        
                    except Exception as e:
                        print(f"Error extracting span {span_idx} from doc {doc_idx}: {e}")
                        continue
            else:
                # Fallback without tokenizer
                return self._extract_sentence_passages(document_text, doc_idx, query)
                        
        except Exception as e:
            print(f"Error processing spans for doc {doc_idx}: {e}")
            return self._extract_sentence_passages(document_text, doc_idx, query)
        
        return passages
    
    def _extract_sentence_passages(self, document_text: str, doc_idx: int, query: str) -> List[Dict]:
        """
        Fallback method: extract sentence-level passages and score by keyword overlap.
        """
        sentences = re.split(r'[.!?]+', document_text)
        query_words = set(query.lower().split())
        
        passages = []
        current_offset = 0
        
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                current_offset += len(sentence) + 1  # +1 for delimiter
                continue
                
            # Simple relevance scoring based on keyword overlap
            sent_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sent_words))
            relevance_score = overlap / max(len(query_words), 1)
            
            if relevance_score > 0:  # Only include sentences with some relevance
                # Find actual position in text
                start_pos = document_text.find(sentence, current_offset)
                end_pos = start_pos + len(sentence) if start_pos != -1 else current_offset + len(sentence)
                
                passages.append({
                    'text': sentence,
                    'core_text': sentence,
                    'relevance_score': relevance_score,
                    'document_id': doc_idx,
                    'span_id': sent_idx,
                    'start_offset': start_pos,
                    'end_offset': end_pos,
                    'highlight_start': start_pos,
                    'highlight_end': end_pos,
                    'query': query
                })
            
            current_offset += len(sentence) + 1  # +1 for delimiter
        
        return passages
    
    def _expand_to_sentence_boundaries(self, text: str, start: int, end: int, 
                                     max_expansion: int = 200) -> str:
        """
        Expand passage to natural sentence boundaries for better readability.
        """
        # Find sentence start (look backwards for sentence delimiters)
        sentence_start = start
        for i in range(max(0, start - max_expansion), start):
            if text[i] in '.!?\n':
                sentence_start = i + 1
                break
        
        # Find sentence end (look forwards for sentence delimiters)  
        sentence_end = end
        for i in range(end, min(len(text), end + max_expansion)):
            if text[i] in '.!?\n':
                sentence_end = i + 1
                break
        
        return text[sentence_start:sentence_end].strip()
    
    def highlight_passages_in_text(self, text: str, passages: List[Dict]) -> str:
        """
        Add HTML highlighting to text based on passage locations.
        """
        if not passages:
            return text
        
        # Sort passages by start position
        sorted_passages = sorted(passages, key=lambda x: x['start_offset'])
        
        highlighted_text = ""
        last_end = 0
        
        for passage in sorted_passages:
            start = passage['highlight_start']
            end = passage['highlight_end']
            
            # Add text before highlight
            highlighted_text += text[last_end:start]
            
            # Add highlighted text
            highlighted_text += f"<mark style='background-color: yellow; font-weight: bold;'>"
            highlighted_text += text[start:end]
            highlighted_text += "</mark>"
            
            last_end = end
        
        # Add remaining text
        highlighted_text += text[last_end:]
        
        return highlighted_text
    
    def format_passages_for_display(self, passages: List[Dict]) -> str:
        """
        Format passages for display in UI with relevance scores and source info.
        """
        if not passages:
            return "No relevant passages found."
        
        formatted_text = f"📋 Top {len(passages)} Relevant Passages:\n"
        formatted_text += "=" * 50 + "\n\n"
        
        for i, passage in enumerate(passages, 1):
            formatted_text += f"🔍 Passage {i} (Relevance Score: {passage['relevance_score']:.3f})\n"
            formatted_text += f"📄 From Document {passage['document_id'] + 1}\n"
            formatted_text += f"📍 Query: {passage['query']}\n"
            formatted_text += "-" * 30 + "\n"
            formatted_text += f'"{passage["text"]}"\n'
            formatted_text += "-" * 30 + "\n\n"
        
        formatted_text += f"💡 Tip: Higher scores indicate more relevant passages.\n"
        formatted_text += f"🎯 These passages were identified using token-level attention analysis."
        
        return formatted_text