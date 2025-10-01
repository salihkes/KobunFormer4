#!/usr/bin/env python3
"""
KobunFormer - Streaming Foreign to Classical Japanese Transformer
Focus: Authentic classical Japanese generation using streaming XML parsing

FEATURES:
- Streaming XML response parsing
- Direct document context integration
- Simplified but effective architecture
- Authentic classical Japanese generation
"""

import os
import sys
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import argparse

# LLM imports
from ollama import chat as ollama_chat
from openai import OpenAI

# Add src to path for RAG import
sys.path.insert(0, 'src')
try:
    from rag import RAGRetriever
    HAS_RAG = True
except ImportError:
    print("RAG system not found, using simplified approach")
    HAS_RAG = False

class LLMClient:
    """A wrapper for different LLM clients like Ollama and OpenAI-compatible servers."""
    def __init__(self, client_type: str = 'ollama', base_url: Optional[str] = None):
        self.client_type = client_type.lower()
        
        if self.client_type == 'openai':
            # For llama.cpp OpenAI-compatible server
            # DISCLAIMER: Change the base_url to your own server's address if needed.
            self.client = OpenAI(base_url=base_url or "http://localhost:8080/v1", api_key="sk-no-key-required")
        elif self.client_type == 'ollama':
            self.client = None # ollama library is used directly
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")

    def chat(self, model: str, messages: List[Dict[str, str]], options: Dict[str, Any], stream: bool):
        if self.client_type == 'ollama':
            return ollama_chat(
                model=model,
                messages=messages,
                options=options,
                stream=stream
            )
        elif self.client_type == 'openai':
            # The 'options' parameter is specific to ollama, so we ignore it.
            # llama.cpp server may support similar parameters, but this is a simple implementation.
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream
            )
            if stream:
                return self._openai_stream_wrapper(response)
            else:
                return {'message': {'content': response.choices[0].message.content}}
    
    def _openai_stream_wrapper(self, stream):
        """Wraps OpenAI stream to match ollama's format."""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield {'message': {'content': content}}

class StreamingXMLParser:
    """Parses streaming XML responses for structured content extraction"""
    
    def __init__(self):
        self.buffer = ""
        self.in_tag = False
        self.current_tag = ""
        self.content = {}
        self.tag_stack = []
        
    def reset(self):
        """Reset parser state for new response"""
        self.buffer = ""
        self.in_tag = False
        self.current_tag = ""
        self.content = {}
        self.tag_stack = []
    
    def feed_chunk(self, chunk: str) -> Dict[str, Any]:
        """Feed a chunk of streaming data and return any completed tags"""
        self.buffer += chunk
        completed_content = {}
        
        # Look for XML tags in buffer
        while True:
            if not self.in_tag:
                # Look for opening tag
                tag_start = self.buffer.find('<')
                if tag_start == -1:
                    break
                    
                tag_end = self.buffer.find('>', tag_start)
                if tag_end == -1:
                    break
                    
                tag = self.buffer[tag_start+1:tag_end]
                
                # Skip closing tags for now
                if tag.startswith('/'):
                    closing_tag = tag[1:]
                    if self.tag_stack and self.tag_stack[-1] == closing_tag:
                        # Complete the tag content
                        completed_content[closing_tag] = self.content.get(closing_tag, "")
                        self.tag_stack.pop()
                        if self.tag_stack:
                            self.current_tag = self.tag_stack[-1]
                        else:
                            self.current_tag = ""
                    self.buffer = self.buffer[tag_end+1:]
                    continue
                
                # Handle opening tag
                self.current_tag = tag
                self.tag_stack.append(tag)
                self.content[tag] = ""
                self.in_tag = True
                self.buffer = self.buffer[tag_end+1:]
                
            else:
                # Look for closing tag or content
                next_tag = self.buffer.find('<')
                if next_tag == -1:
                    # No more tags, add content to current tag
                    if self.current_tag:
                        self.content[self.current_tag] += self.buffer
                    self.buffer = ""
                    break
                else:
                    # Add content before next tag
                    if self.current_tag and next_tag > 0:
                        self.content[self.current_tag] += self.buffer[:next_tag]
                    self.buffer = self.buffer[next_tag:]
                    self.in_tag = False
        
        return completed_content

class KobunAgent:
    """Streaming agent for Foreign to Classical Japanese transformation"""
    
    def __init__(self, llm_client: LLMClient, data_dir: str = "data", works_dir: str = "works", allow_anachronisms: bool = False, use_keigo: bool = False):
        self.llm_client = llm_client
        self.data_dir = Path(data_dir)
        self.works_dir = Path(works_dir)
        self.allow_anachronisms = allow_anachronisms
        self.use_keigo = use_keigo
        self.xml_parser = StreamingXMLParser()
        self.rag_retriever = None
        self.works_rag_retriever = None
        
        # Load linguistic data
        self.linguistic_data = self._load_linguistic_data()
        self.custom_vocab = self._load_custom_vocab("vocab.txt")
        
        # Initialize RAG systems if available
        if HAS_RAG:
            try:
                # Linguistic patterns RAG
                self.rag_retriever = RAGRetriever(
                    use_reranker=True,
                    file_extensions=['.json'],
                    debug=False
                )
                if self.data_dir.exists():
                    self.rag_retriever.load_documents(str(self.data_dir))
                    print(f"Linguistic RAG initialized with {len(self.rag_retriever.document_chunks)} chunks")
                
                # Classical works RAG
                self.works_rag_retriever = RAGRetriever(
                    use_reranker=True,
                    file_extensions=['.txt'],
                    debug=False
                )
                if self.works_dir.exists():
                    self.works_rag_retriever.load_documents(str(self.works_dir))
                    print(f"Works RAG initialized with {len(self.works_rag_retriever.document_chunks)} chunks")
                    
            except Exception as e:
                print(f"RAG initialization failed: {e}")
                self.rag_retriever = None
                self.works_rag_retriever = None
        
        print(f"KobunAgent initialized")
    
    def _load_linguistic_data(self) -> Dict[str, Any]:
        """Load essential linguistic data for context"""
        data = {}
        
        if not self.data_dir.exists():
            print(f"Data directory {self.data_dir} not found")
            return data
        
        # Load all JSON files from the data directory
        for file_path in self.data_dir.glob('*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data[file_path.name] = json.load(f)
                print(f"Loaded {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
        
        return data
    
    def _load_custom_vocab(self, vocab_file: str) -> str:
        """Loads custom vocabulary from a file."""
        vocab_path = Path(vocab_file)
        if not vocab_path.exists():
            print(f"Vocabulary file not found: {vocab_file}")
            return ""
        
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"Loaded custom vocabulary from {vocab_file}")
            return content
        except Exception as e:
            print(f"Error loading vocabulary file {vocab_file}: {e}")
            return ""
    
    def _analyze_japanese_for_queries(self, japanese_text: str) -> List[str]:
        """Analyze Japanese text to generate targeted RAG queries using the LLM."""
        
        query_generation_prompt = f"""
        Analyze the following Modern Japanese text and generate a concise list of 4-5 targeted search queries for a classical Japanese linguistic database. 
        Focus on identifying key nouns, particles, verb endings, and overall grammatical structures that require classical transformation.

        Modern Japanese Text: "{japanese_text}"

        Generate queries that will retrieve information on:
        1. Classical equivalents for modern vocabulary (e.g., '今日' -> '本日').
        2. Correct classical particles and their usage (e.g., 'が' vs. 'の').
        3. Authentic classical verb conjugations and auxiliary verbs (e.g., 'です' -> 'なり', 'である').
        4. Stylistic patterns appropriate for the tone of the text.

        Respond with a list of queries inside <queries> tags, each on a new line.
        Example:
        <queries>
        classical equivalent for 今日
        classical particle usage が vs の
        classical copula なり
        formal greeting classical japanese
        </queries>
        """

        try:
            result = self._stream_with_xml_parsing(query_generation_prompt)
            queries_str = result.get('queries', '').strip()
            if queries_str:
                queries = [q.strip() for q in queries_str.split('\n') if q.strip()]
                print(f"Dynamically generated {len(queries)} RAG queries.")
                return queries[:5]  # Limit to 5 queries
        except Exception as e:
            print(f"Error during dynamic query generation: {e}")

        # Fallback to a simple query if LLM-based generation fails
        return [japanese_text]
    
    def _select_relevant_works(self, japanese_text: str, Foreign_text: str) -> str:
        """Let LLM naturally choose relevant classical works for styling"""
        if not self.works_rag_retriever:
            return ""
        
        # Available works info
        works_info = {
            '竹取物語': 'Taketori Monogatari - prose narrative, simple elegant style, everyday conversations',
            '伊勢物語': 'Ise Monogatari - poetic episodes, romantic themes, concise emotional expression',
            '古今和歌集仮名序': 'Kokinshu Preface - theoretical discourse, refined literary language',
            '源氏物語': 'Genji Monogatari - sophisticated court prose, psychological depth',
            '源氏物語-花散里': 'Genji Monogatari Hanachiru-sato - specific chapter, intimate conversations'
        }
        
        works_selection_prompt = f"""
        Analyze this text to determine the most appropriate classical Japanese work for stylistic guidance:
        
        Foreign: {Foreign_text}
        Japanese: {japanese_text}
        
        Available classical works:
        {chr(10).join([f"- {work}: {desc}" for work, desc in works_info.items()])}
        
        Consider:
        1. Content type (greeting, conversation, gratitude, etc.)
        2. Emotional tone
        3. Formality level
        4. Literary appropriateness
        
        Choose the most naturally fitting work and explain why in 1-2 sentences.
        
        Respond with:
        <selected_work>
        [Work filename exactly as listed]
        </selected_work>
        
        <reasoning>
        [Brief explanation of why this work fits the style needs]
        </reasoning>
        """
        
        try:
            selection_result = self._stream_with_xml_parsing(works_selection_prompt)
            selected_work = selection_result.get('selected_work', '').strip()
            reasoning = selection_result.get('reasoning', '').strip()
            
            if selected_work in works_info:
                print(f"Selected work: {selected_work}")
                print(f"Reasoning: {reasoning}")
                
                # Now retrieve relevant passages from the selected work
                work_queries = [
                    japanese_text,  # Direct match
                    f"greeting conversation {selected_work}",  # Context match
                    f"dialogue {selected_work}",  # Style match
                ]
                
                style_chunks = []
                seen_chunks = set()
                
                for query in work_queries:
                    chunks, _ = self.works_rag_retriever.retrieve(query, top_k=2)
                    for chunk in chunks:
                        chunk_id = f"{chunk.get('document', '')}_{chunk.get('chunk_start', 0)}"
                        if (chunk_id not in seen_chunks and 
                            selected_work in chunk.get('document', '')):
                            seen_chunks.add(chunk_id)
                            style_chunks.append(chunk)
                
                if style_chunks:
                    style_examples = "=== AUTHENTIC CLASSICAL STYLE EXAMPLES ===\n\n"
                    for i, chunk in enumerate(style_chunks[:3]):  # Limit to 3 examples
                        content = chunk.get('output', '')
                        style_examples += f"Example {i+1} from {selected_work}:\n{content[:400]}...\n\n"
                    
                    print(f"Retrieved {len(style_chunks)} style examples from {selected_work}")
                    return style_examples
                    
            else:
                print(f"Selected work '{selected_work}' not found, using general approach")
                
        except Exception as e:
            print(f"Work selection failed: {e}")
        
        return ""
    
    def _get_relevant_context(self, japanese_text: str) -> str:
        """Get relevant linguistic context using multi-query RAG strategy"""
        context = "=== CLASSICAL JAPANESE LINGUISTIC RESOURCES ===\n\n"
        
        # Use enhanced RAG if available
        if self.rag_retriever:
            try:
                # Generate multiple targeted queries
                queries = self._analyze_japanese_for_queries(japanese_text)
                
                all_chunks = []
                seen_chunks = set()
                
                # Execute multiple queries
                for i, query in enumerate(queries):
                    print(f"  Query {i+1}: {query[:50]}...")
                    chunks, indices = self.rag_retriever.retrieve(query, top_k=2)
                    
                    # Deduplicate chunks
                    for chunk in chunks:
                        chunk_id = f"{chunk.get('document', '')}_{chunk.get('chunk_start', 0)}"
                        if chunk_id not in seen_chunks:
                            seen_chunks.add(chunk_id)
                            all_chunks.append(chunk)
                
                if all_chunks:
                    context += "RELEVANT LINGUISTIC PATTERNS:\n"
                    # Prioritize different document types
                    chunks_by_doc = {}
                    for chunk in all_chunks:
                        doc_name = chunk.get('document', 'Unknown')
                        if doc_name not in chunks_by_doc:
                            chunks_by_doc[doc_name] = []
                        chunks_by_doc[doc_name].append(chunk)
                    
                    # Add chunks from different documents for diversity
                    chunk_count = 0
                    for doc_name, doc_chunks in chunks_by_doc.items():
                        if chunk_count >= 5:  # Limit total chunks
                            break
                        for chunk in doc_chunks[:2]:  # Max 2 per document
                            if chunk_count >= 5:
                                break
                            content = chunk.get('output', '')
                            context += f"{chunk_count+1}. From {doc_name}:\n{content[:600]}...\n\n"
                            chunk_count += 1
                    
                    print(f"RAG retrieved {chunk_count} diverse chunks from {len(chunks_by_doc)} documents")
                    return context
                    
            except Exception as e:
                print(f"RAG retrieval failed: {e}")
        
        # Fallback to direct data access
        print("Using fallback direct data access")
        context += "CORE LINGUISTIC PATTERNS:\n"
        
        # Add a sample of loaded JSON data as raw context
        for filename, json_data in self.linguistic_data.items():
            context += f"\n--- Data from {filename} ---\n"
            # Convert dict to string and take a sample to avoid excessive context length
            json_string = json.dumps(json_data, indent=2, ensure_ascii=False)
            context += (json_string[:1000] + '...') if len(json_string) > 1000 else json_string
            context += f"\n--- End of {filename} ---\n"
        
        return context
    
    def _stream_with_xml_parsing(self, prompt: str, model: str = 'qwen3:32b') -> Dict[str, str]:
        """Stream response and parse XML tags in real-time"""
        
        self.xml_parser.reset()
        completed_tags = {}
        
        print(f"AI processing", end="", flush=True)
        
        try:
            response = self.llm_client.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}],
                options={"num_ctx": 32768},
                stream=True
            )
            
            
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    token = chunk['message']['content']
                    print(token, end="", flush=True)
                    
                    # Feed to XML parser
                    new_completions = self.xml_parser.feed_chunk(token)
                    completed_tags.update(new_completions)
            
            print()  # New line after response
            
            # Get any remaining content
            for tag, content in self.xml_parser.content.items():
                if tag not in completed_tags and content.strip():
                    completed_tags[tag] = content.strip()
            
            return completed_tags
            
        except Exception as e:
            print(f"\nStreaming error: {e}")
            return {}
    
    def transform_to_kobun(self, Foreign_text: str) -> Dict[str, Any]:
        """Transform Foreign text to authentic Classical Japanese"""
        
        print(f"\nKOBUN TRANSFORMATION STARTING")
        print(f"Foreign: {Foreign_text}")
        print("=" * 50)
        
        # Step 1: Foreign to Japanese
        print(f"\nStep 1: Foreign -> Japanese")
        
        japanese_prompt = f"""
Translate this Foreign text to natural Modern Japanese:

Foreign: {Foreign_text}

Respond with:
<japanese>
[Japanese translation here]
</japanese>

<rationale>
[Brief explanation of translation choices]
</rationale>
"""
        
        jp_result = self._stream_with_xml_parsing(japanese_prompt)
        japanese_text = jp_result.get('japanese', '').strip()
        
        if not japanese_text:
            return {'error': 'Failed to get Japanese translation'}
        
        print(f"Japanese: {japanese_text}")
        
        # Step 2: Select appropriate classical work for styling
        print(f"\nStep 2: Selecting appropriate classical work")
        style_examples = self._select_relevant_works(japanese_text, Foreign_text)
        
        # Step 3: Get relevant linguistic context
        print(f"\nStep 3: Gathering linguistic context")
        context = self._get_relevant_context(japanese_text)
        print(f"Context prepared ({len(context)} chars)")
        
        # Step 4: Japanese to Classical Japanese
        print(f"\nStep 4: Japanese -> Classical Japanese (Kobun)")
        
        anachronism_rule = ""
        if self.allow_anachronisms:
            anachronism_rule = "\n7. **Anachronism Rule**: If a modern vocabulary word has no direct or suitable classical equivalent, it is acceptable to use the modern word as-is. This is preferable to using an obscure or inappropriate classical term."

        keigo_rule = ""
        if self.use_keigo:
            keigo_rule = """
8. **Keigo (敬語) Application**: This is a priority. Analyze the social context and relationships between individuals (e.g., teacher-student, lord-vassal). You MUST apply the correct form of Keigo:
    - **Sonkeigo (尊敬語)**: Use honorific language for a superior's actions.
    - **Kenjōgo (謙譲語)**: Use humble language for your (or your group's) actions toward a superior.
    - **Teineigo (丁寧語)**: Use polite language to respect the reader/listener.
    The linguistic resources contain specific rules for this. Failure to apply Keigo is incorrect.
"""

        custom_vocab_context = ""
        if self.custom_vocab:
            custom_vocab_context = f"""
CUSTOM VOCABULARY:
Use the 'Use!' column for classical Japanese equivalents of modern terms.
{self.custom_vocab}
"""

        kobun_prompt = f"""
Convert this Modern Japanese to authentic Classical Japanese (Kobun) using the provided resources:

MODERN JAPANESE: {japanese_text}
{custom_vocab_context}
{style_examples}

{context}

Study the authentic classical style examples above. Notice the natural rhythm, concise expression, and genuine classical patterns - NOT artificial elevation.

Transform the Japanese text to match this authentic classical style:
1. Use natural classical vocabulary and particles 
2. Apply genuine classical verb forms and sentence patterns
3. Maintain the original meaning with authentic classical elegance
4. Follow the rhythmic and stylistic patterns shown in the examples
5. Avoid artificially elevated or "philosophical" language unless context demands it
6. Aim for the natural, flowing style of actual classical works{anachronism_rule}{keigo_rule}

Respond with:
<kobun>
[Classical Japanese text here]
</kobun>

<analysis>
[Detailed explanation of classical transformations applied, referencing the style examples]
</analysis>

<authenticity_score>
[Score 1-10 for classical authenticity and natural style]
</authenticity_score>
"""
        
        kobun_result = self._stream_with_xml_parsing(kobun_prompt)
        kobun_text = kobun_result.get('kobun', '').strip()
        analysis = kobun_result.get('analysis', '').strip()
        authenticity = kobun_result.get('authenticity_score', '').strip()
        
        # Parse authenticity score
        try:
            auth_score = float(re.findall(r'\d+\.?\d*', authenticity)[0]) if authenticity else 0.0
        except:
            auth_score = 0.0
            
        # Step 5: Validation
        validation_data = self._validate_kobun(kobun_text, japanese_text, analysis)

        # Final result
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input': {
                'Foreign': Foreign_text
            },
            'output': {
                'japanese': japanese_text,
                'kobun': kobun_text,
                'analysis': analysis,
                'authenticity_score': auth_score,
                'validation': validation_data
            },
            'success': bool(kobun_text and validation_data.get('score', 0) >= 5)
        }
        
        print(f"\nTRANSFORMATION COMPLETE")
        print("=" * 30)
        print(f"Foreign: {Foreign_text}")
        print(f"Japanese: {japanese_text}")
        print(f"Kobun: {kobun_text}")
        print(f"Authenticity: {auth_score}/10")
        
        return result
    
    def transform_japanese_to_kobun(self, japanese_text: str) -> Dict[str, Any]:
        """Transform Japanese text directly to Classical Japanese (skipping Foreign step)"""
        
        print(f"\nKOBUN TRANSFORMATION STARTING (Japanese Input)")
        print(f"Japanese: {japanese_text}")
        print("=" * 50)
        
        # Step 1: Select appropriate classical work for styling
        print(f"\nStep 1: Selecting appropriate classical work")
        # For Japanese input, we don't have Foreign text, so we pass empty string
        style_examples = self._select_relevant_works(japanese_text, "")
        
        # Step 2: Get relevant linguistic context
        print(f"\nStep 2: Gathering linguistic context")
        context = self._get_relevant_context(japanese_text)
        print(f"Context prepared ({len(context)} chars)")
        
        # Step 3: Japanese to Classical Japanese
        print(f"\nStep 3: Japanese -> Classical Japanese (Kobun)")
        
        anachronism_rule = ""
        if self.allow_anachronisms:
            anachronism_rule = "\n7. **Anachronism Rule**: If a modern vocabulary word has no direct or suitable classical equivalent, it is acceptable to use the modern word as-is. This is preferable to using an obscure or inappropriate classical term."
        
        keigo_rule = ""
        if self.use_keigo:
            keigo_rule = """
8. **Keigo (敬語) Application**: This is a priority. Analyze the social context and relationships between individuals (e.g., teacher-student, lord-vassal). You MUST apply the correct form of Keigo:
    - **Sonkeigo (尊敬語)**: Use honorific language for a superior's actions.
    - **Kenjōgo (謙譲語)**: Use humble language for your (or your group's) actions toward a superior.
    - **Teineigo (丁寧語)**: Use polite language to respect the reader/listener.
    The linguistic resources contain specific rules for this. Failure to apply Keigo is incorrect.
"""

        custom_vocab_context = ""
        if self.custom_vocab:
            custom_vocab_context = f"""
CUSTOM VOCABULARY:
Use the 'Use!' column for classical Japanese equivalents of modern terms.
{self.custom_vocab}
"""
        
        kobun_prompt = f"""
Convert this Modern Japanese to authentic Classical Japanese (Kobun) using the provided resources:

MODERN JAPANESE: {japanese_text}
{custom_vocab_context}
{style_examples}

{context}

Study the authentic classical style examples above. Notice the natural rhythm, concise expression, and genuine classical patterns - NOT artificial elevation.

Transform the Japanese text to match this authentic classical style:
1. Use natural classical vocabulary and particles 
2. Apply genuine classical verb forms and sentence patterns
3. Maintain the original meaning with authentic classical elegance
4. Follow the rhythmic and stylistic patterns shown in the examples
5. Avoid artificially elevated or "philosophical" language unless context demands it
6. Aim for the natural, flowing style of actual classical works{anachronism_rule}{keigo_rule}

Respond with:
<kobun>
[Classical Japanese text here]
</kobun>

<analysis>
[Detailed explanation of classical transformations applied, referencing the style examples]
</analysis>

<authenticity_score>
[Score 1-10 for classical authenticity and natural style]
</authenticity_score>
"""
        
        kobun_result = self._stream_with_xml_parsing(kobun_prompt)
        kobun_text = kobun_result.get('kobun', '').strip()
        analysis = kobun_result.get('analysis', '').strip()
        authenticity = kobun_result.get('authenticity_score', '').strip()
        
        # Parse authenticity score
        try:
            auth_score = float(re.findall(r'\d+\.?\d*', authenticity)[0]) if authenticity else 0.0
        except:
            auth_score = 0.0
            
        # Step 4: Validation
        validation_data = self._validate_kobun(kobun_text, japanese_text, analysis)

        # Final result
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input': {
                'japanese': japanese_text
            },
            'output': {
                'kobun': kobun_text,
                'analysis': analysis,
                'authenticity_score': auth_score,
                'validation': validation_data
            },
            'success': bool(kobun_text and validation_data.get('score', 0) >= 5)
        }
        
        print(f"\nTRANSFORMATION COMPLETE")
        print("=" * 30)
        print(f"Japanese: {japanese_text}")
        print(f"Kobun: {kobun_text}")
        print(f"Authenticity: {auth_score}/10")
        
        return result
    
    def transform_chinese_to_kundoku(self, chinese_text: str) -> Dict[str, Any]:
        """Transform Classical Chinese text to a Kundoku reading."""
        
        print(f"\nKUNDOKU TRANSFORMATION STARTING (Classical Chinese Input)")
        print(f"Classical Chinese: {chinese_text}")
        print("=" * 50)
        
        # Step 1: Preliminary translation for context gathering
        print(f"\nStep 1: Preliminary translation for context gathering")
        japanese_context_prompt = f"""
Translate this Classical Chinese text to Modern Japanese. This is for context analysis only.

Classical Chinese: {chinese_text}

Respond with:
<japanese_context>
[Modern Japanese translation]
</japanese_context>
"""
        jp_context_result = self._stream_with_xml_parsing(japanese_context_prompt)
        japanese_context_text = jp_context_result.get('japanese_context', '').strip()

        if not japanese_context_text:
            japanese_context_text = chinese_text # fallback

        print(f"Japanese Context: {japanese_context_text}")

        # Step 2: Select appropriate classical work for styling
        print(f"\nStep 2: Selecting appropriate classical work")
        style_examples = self._select_relevant_works(japanese_context_text, chinese_text)
        
        # Step 3: Get relevant linguistic context
        print(f"\nStep 3: Gathering linguistic context")
        context = self._get_relevant_context(japanese_context_text)
        print(f"Context prepared ({len(context)} chars)")
        
        # Step 4: Classical Chinese to Kundoku
        print(f"\nStep 4: Classical Chinese -> Kundoku")
        
        kundoku_prompt = f"""
You are an expert in transforming Classical Chinese texts into Japanese Kundoku (訓読).
Your task is to convert the given Classical Chinese text into a natural-sounding Classical Japanese reading, following the principles of Kundoku.

PRINCIPLES OF KUNDOKU:
1.  **Preserve Kanji**: Keep all original Chinese characters.
2.  **Reorder**: Rearrange the characters to follow Classical Japanese grammar and syntax. This includes using kaeriten (返り点) logic implicitly.
3.  **Add Okurigana**: Add hiragana suffixes (okurigana) to indicate conjugations and grammatical functions.
4.  **Add Particles**: Insert Classical Japanese particles (助詞) and auxiliary verbs (助動詞) where necessary to form a coherent sentence.
5.  **Authenticity**: The final output should read like authentic Classical Japanese (Kobun), not a stiff, literal translation.

CLASSICAL CHINESE TEXT:
{chinese_text}

RESOURCES:
{style_examples}
{context}

INSTRUCTIONS:
1.  Analyze the Classical Chinese text.
2.  Apply the principles of Kundoku to transform it into a Classical Japanese reading.
3.  Use the provided style examples and linguistic resources to ensure authenticity.
4.  The final output must be a valid and natural-sounding Classical Japanese sentence.

Respond with:
<kundoku>
[The Kundoku reading in Classical Japanese]
</kundoku>

<analysis>
[A detailed explanation of the transformation, including the reordering of characters, and the reasoning for added okurigana, particles, and auxiliary verbs.]
</analysis>

<authenticity_score>
[Score from 1-10 for the authenticity and correctness of the Kundoku reading.]
</authenticity_score>
"""
        
        kundoku_result = self._stream_with_xml_parsing(kundoku_prompt)
        kundoku_text = kundoku_result.get('kundoku', '').strip()
        analysis = kundoku_result.get('analysis', '').strip()
        authenticity = kundoku_result.get('authenticity_score', '').strip()
        
        try:
            auth_score = float(re.findall(r'\d+\.?\d*', authenticity)[0]) if authenticity else 0.0
        except:
            auth_score = 0.0
            
        # Step 5: Validation
        # NOTE: _validate_kundoku needs to be implemented
        validation_data = self._validate_kundoku(kundoku_text, chinese_text, analysis)

        # Final result
        result = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input': {
                'classical_chinese': chinese_text
            },
            'output': {
                'kundoku': kundoku_text,
                'analysis': analysis,
                'authenticity_score': auth_score,
                'validation': validation_data
            },
            'success': bool(kundoku_text and validation_data.get('score', 0) >= 5) # Changed from >= 5 to >= 0 for now
        }
        
        print(f"\nTRANSFORMATION COMPLETE")
        print("=" * 30)
        print(f"Classical Chinese: {chinese_text}")
        print(f"Kundoku: {kundoku_text}")
        print(f"Authenticity: {auth_score}/10")
        
        return result
    
    def _validate_kobun(self, kobun_text: str, japanese_text: str, analysis: str) -> Dict[str, Any]:
        """Validate the generated Kobun text for authenticity and correctness."""
        print("\nStep 5: Validating Kobun authenticity")

        custom_vocab_context = ""
        if self.custom_vocab:
            custom_vocab_context = f"""
CUSTOM VOCABULARY TO ENFORCE:
The following vocabulary should be used for modern terms.
{self.custom_vocab}
"""

        anachronism_notice = ""
        if self.allow_anachronisms:
            anachronism_notice = """
**IMPORTANT NOTE ON ANACHRONISMS**: Anachronistic vocabulary (using a modern word when no classical equivalent exists) was permitted during generation. Do not penalize the score or feedback for the appropriate use of modern words. Instead, confirm if their use was necessary and well-integrated.
"""

        validation_prompt = f"""
        Please validate the following Classical Japanese (Kobun) text.
        {anachronism_notice}
        Modern Japanese (for context): "{japanese_text}"
        Generated Kobun: "{kobun_text}"
        Transformation Analysis: "{analysis}"
        {custom_vocab_context}
        Perform the following checks:
        1.  **Grammatical Correctness**: Are particles, verb conjugations, and sentence structures correct for authentic Classical Japanese?
        2.  **Authenticity**: Does the text reflect a genuine classical style, or does it feel artificial?
        3.  **Faithfulness**: Does the Kobun text retain the meaning and nuance of the Modern Japanese original?
        4.  **Specific Rules**: Check for the correct application of specific grammatical rules, such as Kakari-Musubi (係り結び), if applicable. Mention whether it was used correctly or if there were missed opportunities.
        5.  **Custom Vocabulary**: If a custom vocabulary was provided, verify that it was used correctly for the relevant modern terms.

        Provide a validation score from 1 to 10 and detailed feedback.

        Respond in the following format:
        <validation_score>[Score from 1-10]</validation_score>
        <validation_feedback>[Detailed feedback on the four points above]</validation_feedback>
        """

        validation_result = self._stream_with_xml_parsing(validation_prompt)
        
        try:
            score_str = validation_result.get('validation_score', '0').strip()
            score = float(re.findall(r'\d+\.?\d*', score_str)[0]) if score_str else 0.0
        except (ValueError, IndexError):
            score = 0.0
            
        feedback = validation_result.get('validation_feedback', 'No feedback provided.').strip()

        print(f"Validation Score: {score}/10")
        print(f"Validation Feedback: {feedback}")

        return {"score": score, "feedback": feedback}

    def _validate_kundoku(self, kundoku_text: str, chinese_text: str, analysis: str) -> Dict[str, Any]:
        """Validate the generated Kundoku text for correctness."""
        print("\nStep 5: Validating Kundoku correctness")

        custom_vocab_context = ""
        if self.custom_vocab:
            custom_vocab_context = f"""
CUSTOM VOCABULARY TO ENFORCE:
The following vocabulary should be used for modern terms.
{self.custom_vocab}
"""

        validation_prompt = f"""
        Please validate the following Japanese Kundoku reading of a Classical Chinese text.

        Original Classical Chinese: "{chinese_text}"
        Generated Kundoku Reading: "{kundoku_text}"
        Transformation Analysis: "{analysis}"
        {custom_vocab_context}
        Perform the following checks:
        1.  **Grammatical Correctness**: Is the Kundoku reading grammatically correct as Classical Japanese? Check particles, verb conjugations, and word order.
        2.  **Faithfulness to Original**: Does the reading accurately represent the meaning of the Classical Chinese text?
        3.  **Kundoku Rules**: Verify that the transformation followed Kundoku principles:
            - Were all original kanji preserved?
            - Is the reordering of characters logical (as if following kaeriten)?
            - Are okurigana, particles, and auxiliary verbs added correctly and appropriately?
        4.  **Authenticity**: Does the final text sound like natural, authentic Classical Japanese?
        5.  **Custom Vocabulary**: If a custom vocabulary was provided, ensure it was used if applicable.

        Provide a validation score from 1 to 10 and detailed feedback.

        Respond in the following format:
        <validation_score>[Score from 1-10]</validation_score>
        <validation_feedback>[Detailed feedback on the four points above]</validation_feedback>
        """

        validation_result = self._stream_with_xml_parsing(validation_prompt)
        
        try:
            score_str = validation_result.get('validation_score', '0').strip()
            score = float(re.findall(r'\d+\.?\d*', score_str)[0]) if score_str else 0.0
        except (ValueError, IndexError):
            score = 0.0
            
        feedback = validation_result.get('validation_feedback', 'No feedback provided.').strip()

        print(f"Validation Score: {score}/10")
        print(f"Validation Feedback: {feedback}")

        return {"score": score, "feedback": feedback}

    def save_result(self, result: Dict[str, Any], filename: str = None) -> str:
        """Save transformation result to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kobun_transform_{timestamp}.json"
        
        output_dir = Path("transformations")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"Result saved: {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Error saving result: {e}")
            return ""
    
    def run_interactive(self, japanese_mode: bool = False, chinese_mode: bool = False):
        """Run interactive transformation mode"""
        print(f"\nKOBUNFORMER - STREAMING XML AGENT")
        if japanese_mode:
            print("Japanese -> Classical Japanese")
        elif chinese_mode:
            print("Classical Chinese -> Kundoku")
        else:
            print("Foreign -> Japanese -> Classical Japanese")
        print("=" * 50)
        
        while True:
            try:
                if japanese_mode:
                    print(f"\nEnter Japanese text to transform:")
                    input_text = input("Japanese: ").strip()
                elif chinese_mode:
                    print(f"\nEnter Classical Chinese text to transform:")
                    input_text = input("Classical Chinese: ").strip()
                else:
                    print(f"\nEnter Foreign text to transform:")
                    input_text = input("Foreign: ").strip()
                
                if not input_text:
                    print("Empty input")
                    continue
                
                if input_text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                # Transform
                if japanese_mode:
                    result = self.transform_japanese_to_kobun(input_text)
                elif chinese_mode:
                    result = self.transform_chinese_to_kundoku(input_text)
                else:
                    result = self.transform_to_kobun(input_text)
                
                if result.get('success'):
                    # Save result
                    saved_path = self.save_result(result)
                    if saved_path:
                        print(f"Result saved: {saved_path}")
                else:
                    print("Transformation failed")
                
                # Continue?
                continue_choice = input(f"\nTransform another text? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
                
            except KeyboardInterrupt:
                print(f"\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="KobunFormer - Streaming Foreign to Classical Japanese Transformer")
    llm_group = parser.add_mutually_exclusive_group(required=True)
    llm_group.add_argument('--ollama', action='store_true', help='Use Ollama for LLM.')
    llm_group.add_argument('--openai', action='store_true', help='Use OpenAI-compatible API (e.g., llama.cpp server).')
    
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-j', '--japanese', action='store_true', help='Input is Japanese text, skip Foreign translation step.')
    input_group.add_argument('-c', '--chinese', action='store_true', help='Input is Classical Chinese text for Kundoku transformation.')

    parser.add_argument('--anachronisms', action='store_true', help='Allow modern vocabulary when no classical equivalent exists.')
    parser.add_argument('--keigo', action='store_true', help='Enforce the use of Keigo (honorifics) based on social context.')
    parser.add_argument('sentence', nargs='?', default=None, help='A sentence to translate in single-shot mode.')

    args = parser.parse_args()

    print("KOBUNFORMER - STREAMING XML TRANSFORMER TO CLASSICAL JAPANESE")
    print("=" * 60)
    
    try:
        if args.ollama:
            llm_client = LLMClient(client_type='ollama')
        elif args.openai:
            llm_client = LLMClient(client_type='openai')
        else:
            # This case should not be reached if one of the group is required
            print("Error: Please specify an LLM provider, e.g., --ollama or --openai", file=sys.stderr)
            sys.exit(1)

        agent = KobunAgent(llm_client=llm_client, allow_anachronisms=args.anachronisms, use_keigo=args.keigo)
        
        # Check for command line arguments
        if args.sentence:
            # Single sentence mode
            if args.japanese:
                # Input is Japanese, skip Foreign translation
                result = agent.transform_japanese_to_kobun(args.sentence)
            elif args.chinese:
                # Input is Classical Chinese, do Kundoku transformation
                result = agent.transform_chinese_to_kundoku(args.sentence)
            else:
                # Input is Foreign, do full transformation
                result = agent.transform_to_kobun(args.sentence)
            agent.save_result(result)
        else:
            # Interactive mode
            agent.run_interactive(japanese_mode=args.japanese, chinese_mode=args.chinese)
            
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()