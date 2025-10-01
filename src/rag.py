import os
import numpy as np
import json
import re
from typing import List, Dict, Any, Set, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

##RAG implementation
##IMPORTANT NOTE TO AI ASSISTANTS: DO NOT MODIFY THIS CODE OR NEURAL RERANKER IMPLEMENTATION
##This implementation is intentional and authorized by the user.

# Try to import necessary libraries
try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Transformers not found. Will fall back to TF-IDF for retrieval.")

# Try to import FlagEmbedding for BGE-M3
try:
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
    HAS_FLAG_EMBEDDING = True
except ImportError:
    HAS_FLAG_EMBEDDING = False
    print("FlagEmbedding not found. Will use TF-IDF fallback.")

class DocumentProcessor:
    """Handles document loading and chunking"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Define excluded directories and file types
        self.excluded_dirs = ['audio_cache', '.rag_cache', '__pycache__']
        self.excluded_file_types = ['.wav', '.mp3', '.ogg', '.mp4']
    
    def load_document(self, file_path: str) -> str:
        """Load a document from file path"""
        try:
            # Skip file if it has an excluded extension
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext in self.excluded_file_types:
                print(f"Skipping excluded file type: {file_path}")
                return ""
                
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading document {file_path}: {str(e)}")
            return ""
    
    def chunk_document(self, document: str, document_title: str = "") -> List[Dict[str, str]]:
        """Break a document into overlapping chunks with smart boundary detection"""
        if not document:
            return []
        
        chunks = []
        
        # Define boundary markers to improve chunk divisions
        paragraph_markers = ["\n\n", "\n"]
        sentence_markers = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
        
        start = 0
        while start < len(document):
            # Calculate initial end position
            end = min(start + self.chunk_size, len(document))
            
            # If we're not at the end of the document, look for a good break point
            if end < len(document):
                # Try to find paragraph breaks first (strongest division)
                best_break = None
                for marker in paragraph_markers:
                    # Look for marker within a reasonable range around ideal chunk end
                    search_start = max(start + (self.chunk_size // 2), start)  # At least half of desired chunk size
                    search_end = min(end + 200, len(document))  # Look a bit beyond the nominal end
                    
                    # Find the last occurrence of the marker in our search range
                    marker_pos = document.rfind(marker, search_start, search_end)
                    
                    if marker_pos != -1:
                        # Add marker length to position to include the marker in the chunk
                        marker_end = marker_pos + len(marker)
                        best_break = marker_end
                        break
                
                # If no paragraph breaks, try sentence breaks
                if best_break is None:
                    for marker in sentence_markers:
                        search_start = max(start + (self.chunk_size // 2), start)
                        search_end = min(end + 100, len(document))
                        
                        marker_pos = document.rfind(marker, search_start, search_end)
                        
                        if marker_pos != -1:
                            marker_end = marker_pos + len(marker)
                            best_break = marker_end
                            break
                
                # If still no good break point, use the maximum chunk size
                if best_break:
                    end = best_break
            
            # Extract chunk text
            chunk_text = document[start:end].strip()
            
            # Add chunk with metadata
            if chunk_text:
                chunks.append({
                    "input": f"Document: {document_title} (Chunk)",
                    "output": chunk_text,
                    "document": document_title,
                    "chunk_start": start,
                    "chunk_end": end
                })
            
            # Move to next chunk with overlap
            start = end - self.chunk_overlap if end < len(document) else end
        
        return chunks

    def process_directory(self, directory: str, file_extensions: List[str] = None) -> List[Dict[str, str]]:
        """Process all documents in a directory with specified extensions"""
        if file_extensions is None:
            file_extensions = ['.txt', '.md', '.json', '.yaml', '.yml']
            
        document_chunks = []
        doc_sources = {}
        
        # Process each file in the directory
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            # Skip if current directory contains any excluded directory name
            if any(excluded in root for excluded in self.excluded_dirs):
                continue
                
            for file in files:
                # Skip files starting with "._" (macOS resource files)
                if file.startswith("._"):
                    continue
                    
                # Skip files with excluded extensions
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in self.excluded_file_types:
                    continue
                    
                # Check if file has an allowed extension
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    document_title = os.path.splitext(file)[0]
                    
                    # Load and chunk document
                    document_text = self.load_document(file_path)
                    chunks = self.chunk_document(document_text, document_title)
                    document_chunks.extend(chunks)
                    
                    # Track document sources
                    if chunks:
                        doc_sources[document_title] = len(chunks)
        
        # Log loaded document sources
        if doc_sources:
            print("Document sources loaded:")
            for source, count in doc_sources.items():
                print(f"- {source}: {count} chunks")
        
        return document_chunks

class SimpleTokenizer:
    """A basic tokenizer for text"""
    
    def __init__(self):
        self.pattern = re.compile(r'\b\w+\b')
    
    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens"""
        return self.pattern.findall(text.lower())

class RAGRetriever:
    """Handles document embedding and retrieval"""
    
    def __init__(self, use_reranker=True, file_extensions=None, debug=False):
        self.document_processor = DocumentProcessor()
        self.document_chunks = []
        self.use_reranker = use_reranker
        self.file_extensions = file_extensions or ['.txt', '.md', '.json']  # Default file extensions to process
        self.debug = debug  # Debug mode flag
        
        # Try to initialize neural models
        self.use_neural = HAS_FLAG_EMBEDDING
        if self.use_neural:
            try:
                print("Loading BGE-M3 embedding model...")
                self.embedding_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
                
                if self.use_reranker and HAS_FLAG_EMBEDDING:
                    print("Loading BGE-reranker model...")
                    self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
                else:
                    self.use_reranker = False
                    
                # Initialize empty cache
                self.cached_embeddings = None
            except Exception as e:
                print(f"Error initializing neural models: {str(e)}")
                self.use_neural = False
        
        # Initialize TF-IDF as fallback
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            min_df=2,  # Minimum document frequency
            max_df=0.95  # Maximum document frequency
        )
        self.document_vectors = None
        
    def load_documents(self, directory: str, file_extensions: List[str] = None) -> None:
        """Load and process documents from a directory"""
        print(f"Processing documents from directory: {directory}")
        # Use provided extensions or default from the class
        extensions_to_use = file_extensions or self.file_extensions
        print(f"Using file extensions: {extensions_to_use}")
        self.document_chunks = self.document_processor.process_directory(directory, extensions_to_use)
        print(f"Loaded {len(self.document_chunks)} document chunks")
        
        # Generate a cache path for this directory
        cache_dir = os.path.join(os.path.dirname(directory), '.rag_cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create a hash of the directory to use as cache key
        dir_name = os.path.basename(os.path.normpath(directory))
        cache_key = f"{dir_name}_{len(self.document_chunks)}"
        cache_path = os.path.join(cache_dir, f"{cache_key}.npz")
        
        # Try to load from cache first
        if self.use_neural and os.path.exists(cache_path):
            print(f"Trying to load embeddings from cache: {cache_path}")
            if self._load_embeddings_from_cache(cache_path):
                print("Successfully loaded embeddings from cache")
                return
        
        # Compute embeddings based on available methods if no cache or cache failed
        if self.use_neural:
            self._compute_dataset_embeddings()
            # Save to cache if embeddings were successfully computed
            if self.cached_embeddings is not None:
                self._save_embeddings_to_cache(cache_path)
        else:
            self._compute_tfidf()
    
    def _save_embeddings_to_cache(self, cache_path: str) -> bool:
        """Save embeddings to cache file"""
        try:
            if self.cached_embeddings is None or len(self.cached_embeddings) == 0:
                return False
                
            # Save embeddings as numpy array
            embeddings_array = np.array(self.cached_embeddings)
            
            # Save document chunk metadata as JSON
            chunks_metadata = []
            for chunk in self.document_chunks:
                # Only save essential metadata to reduce file size
                chunks_metadata.append({
                    "document": chunk.get("document", ""),
                    "chunk_start": chunk.get("chunk_start", 0),
                    "chunk_end": chunk.get("chunk_end", 0),
                    "input": chunk.get("input", ""),
                    "output": chunk.get("output", "")
                })
            
            # Save both the embeddings and the chunks metadata
            np.savez_compressed(
                cache_path,
                embeddings=embeddings_array,
                chunks_json=json.dumps(chunks_metadata)
            )
            
            print(f"Saved embeddings to cache: {cache_path}")
            return True
        except Exception as e:
            print(f"Error saving embeddings to cache: {str(e)}")
            return False
    
    def _load_embeddings_from_cache(self, cache_path: str) -> bool:
        """Load embeddings from cache file"""
        try:
            # Load the npz file
            cached_data = np.load(cache_path, allow_pickle=True)
            
            # Load the embeddings
            embeddings_array = cached_data['embeddings']
            self.cached_embeddings = [embeddings_array[i] for i in range(len(embeddings_array))]
            
            # Load the chunks metadata
            chunks_json = cached_data['chunks_json'].item()
            self.document_chunks = json.loads(chunks_json)
            
            print(f"Loaded {len(self.cached_embeddings)} embeddings and {len(self.document_chunks)} chunks from cache")
            return True
        except Exception as e:
            print(f"Error loading embeddings from cache: {str(e)}")
            return False
    
    def _compute_dataset_embeddings(self) -> None:
        """Compute embeddings for all document chunks using BGE-M3"""
        if not self.document_chunks:
            self.cached_embeddings = None
            return
        
        texts = [item["output"] for item in self.document_chunks]
        print(f"Computing embeddings for {len(texts)} document chunks...")
        
        try:
            # Using batched approach to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                # Get embeddings for the batch
                batch_embeddings = self.embedding_model.encode(batch_texts, return_dense=True)
                
                # Store embeddings
                if isinstance(batch_embeddings, dict) and "dense_vecs" in batch_embeddings:
                    all_embeddings.extend(batch_embeddings["dense_vecs"])
                    
            # Verify we have valid embeddings
            if len(all_embeddings) == len(texts):
                self.cached_embeddings = all_embeddings
                print(f"Successfully computed embeddings for {len(texts)} document chunks")
            else:
                print(f"Warning: Expected {len(texts)} embeddings but got {len(all_embeddings)}. Falling back to TF-IDF.")
                self.use_neural = False
                self._compute_tfidf()
        except Exception as e:
            print(f"Error computing embeddings: {str(e)}")
            print("Falling back to TF-IDF method.")
            self.use_neural = False
            self._compute_tfidf()
    
    def _compute_tfidf(self) -> None:
        """Compute TF-IDF vectors for all document chunks (fallback method)"""
        if not self.document_chunks:
            self.document_vectors = None
            return
            
        # Extract texts for vectorization
        texts = [f"{item['input']} {item['output']}" for item in self.document_chunks]
        
        # Handle edge case with only one document or very short documents
        if len(texts) == 1:
            # Use a simpler approach for a single document
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                min_df=1,  # Allow terms that appear only once
                max_df=1.0  # Allow terms that appear in all documents
            )
        
        # Compute TF-IDF vectors
        try:
            self.document_vectors = self.vectorizer.fit_transform(texts)
            print(f"Computed TF-IDF vectors for {len(self.document_chunks)} chunks")
        except ValueError as e:
            # Handle case with no features
            print(f"Warning: {str(e)}. Using simple word count for retrieval.")
            # Fall back to a simpler approach
            tokenizer = SimpleTokenizer()
            word_counts = []
            for text in texts:
                tokens = tokenizer.tokenize(text)
                word_counts.append({word: tokens.count(word) for word in set(tokens)})
            self.document_vectors = word_counts
    
    def retrieve(self, query: str, top_k: int = 3, 
                already_retrieved: Optional[Set[int]] = None) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Retrieve relevant document chunks for a query
        
        Args:
            query: The search query
            top_k: Number of results to return
            already_retrieved: Set of indices to exclude from results
            
        Returns:
            Tuple of (retrieved chunks, their indices)
        """
        if not self.document_chunks:
            return [], []
        
        # Initialize set if none provided
        if already_retrieved is None:
            already_retrieved = set()
        
        if self.debug:
            print(f"\n[DEBUG] Retrieving for query: '{query}'")
            
        if self.use_neural and self.cached_embeddings:
            chunks, indices = self._retrieve_neural(query, top_k, already_retrieved)
        else:
            chunks, indices = self._retrieve_tfidf(query, top_k, already_retrieved)
            
        # Print retrieved chunks in debug mode
        if self.debug and chunks:
            print(f"\n[DEBUG] Retrieved {len(chunks)} chunks:")
            for i, chunk in enumerate(chunks):
                doc_title = chunk.get("document", "Untitled Document")
                score = getattr(chunk, 'score', 'N/A')
                print(f"\n--- Chunk {i+1} (Doc: {doc_title}, Score: {score}) ---")
                # Print a preview of the chunk content (first 200 chars)
                content = chunk.get("output", "")
                print(f"{content[:200]}..." if len(content) > 200 else content)
            print("\n[DEBUG] End of retrieved chunks")
            
        return chunks, indices
    
    def _retrieve_neural(self, query: str, top_k: int, already_retrieved: Set[int]) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Retrieve using neural embeddings and optional reranking"""
        try:
            if self.debug:
                print(f"[DEBUG] Using neural retrieval with {'reranking' if self.use_reranker else 'no reranking'}")
                
            # Get embedding for the query
            query_embedding = self.embedding_model.encode([query], return_dense=True)["dense_vecs"][0]
            
            # Compute similarity with all document embeddings
            scores = []
            for i, doc_embedding in enumerate(self.cached_embeddings):
                if i not in already_retrieved:
                    # Compute cosine similarity
                    similarity = np.dot(query_embedding, doc_embedding)
                    scores.append((similarity, i))
            
            # Sort by score and get top candidates
            scores.sort(reverse=True)
            candidate_indices = [idx for _, idx in scores[:min(top_k*3, len(scores))]]
            
            # If using reranker, rerank the candidates
            if self.use_reranker and len(candidate_indices) > top_k:
                # Create query-document pairs
                pairs = []
                for idx in candidate_indices:
                    pairs.append((query, self.document_chunks[idx]["output"]))
                
                # Rerank using the reranker
                rerank_scores = self.reranker.compute_score(pairs)
                
                # Sort by reranker score
                reranked = [(score, idx) for score, idx in zip(rerank_scores, candidate_indices)]
                reranked.sort(reverse=True)
                
                # Get top_k indices after reranking
                final_indices = [idx for _, idx in reranked[:top_k]]
                
                if self.debug:
                    print(f"[DEBUG] Reranked {len(candidate_indices)} candidates to {len(final_indices)} final results")
            else:
                final_indices = candidate_indices[:top_k]
            
            # Get the corresponding chunks
            final_chunks = [self.document_chunks[idx] for idx in final_indices]
            
            # Add scores to chunks for debugging
            if self.debug:
                if self.use_reranker and len(candidate_indices) > top_k:
                    for i, (score, _) in enumerate(reranked[:top_k]):
                        final_chunks[i]['score'] = round(float(score), 4)
                else:
                    for i, (score, _) in enumerate(scores[:top_k]):
                        final_chunks[i]['score'] = round(float(score), 4)
            
            return final_chunks, final_indices
        except Exception as e:
            print(f"Error during neural retrieval: {str(e)}")
            # Fall back to TF-IDF
            return self._retrieve_tfidf(query, top_k, already_retrieved)
    
    def _retrieve_tfidf(self, query: str, top_k: int, already_retrieved: Set[int]) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Retrieve using TF-IDF vectors"""
        if self.debug:
            print(f"[DEBUG] Using TF-IDF retrieval")
            
        if isinstance(self.document_vectors, list):
            # Using the simple word count approach
            tokenizer = SimpleTokenizer()
            query_tokens = tokenizer.tokenize(query)
            query_counts = {word: query_tokens.count(word) for word in set(query_tokens)}
            
            # Compute similarity with all documents
            similarities = []
            for idx, doc_counts in enumerate(self.document_vectors):
                if idx in already_retrieved:
                    continue
                
                # Compute simple dot product similarity
                score = 0
                for word, count in query_counts.items():
                    if word in doc_counts:
                        score += count * doc_counts[word]
                
                similarities.append((score, idx))
        else:
            # Using scikit-learn's TF-IDF and cosine similarity
            query_vector = self.vectorizer.transform([query])
            
            # Compute similarities for all documents
            similarity_matrix = cosine_similarity(query_vector, self.document_vectors)[0]
            
            # Convert to list of tuples (similarity, index)
            similarities = [(similarity_matrix[idx], idx) for idx in range(len(self.document_chunks)) 
                         if idx not in already_retrieved]
        
        # If all contexts have been retrieved already, return empty
        if not similarities:
            return [], []
        
        # Sort by similarity and take top_k
        similarities.sort(reverse=True)
        top_indices = [idx for _, idx in similarities[:top_k]]
        
        # Get top contexts
        top_contexts = [self.document_chunks[idx] for idx in top_indices]
        
        return top_contexts, top_indices
    
    def format_retrieved_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks for inclusion in system message"""
        if not chunks:
            return ""
        
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            doc_title = chunk.get("document", "Untitled Document")
            chunk_text = chunk.get("output", "")
            
            formatted_chunks.append(
                f"Document: {doc_title} (Chunk {i+1})\n{chunk_text}"
            )
        
        formatted_result = "<rag_results>\n" + "\n\n".join(formatted_chunks) + "\n</rag_results>"
        
        # In debug mode, also print the formatted result
        if self.debug:
            print("\n[DEBUG] Formatted RAG Results:")
            print(formatted_result)
            print("[DEBUG] End of formatted results\n")
            
        return formatted_result 