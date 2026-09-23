from transformers import logging
logging.set_verbosity_error()

from typing import Union, Dict, Any
from knowledge_base import KnowledgeBase
from websearch import WebSearch
from chunker import TextChunker
from retriever import SemanticReranker
from generator import TransformerGenerator

# -------------------------------------------------------------
# Modular Local RAG Pipeline: Ingestion -> Chunking -> Reranking -> Generation
# -------------------------------------------------------------
class QABot:
    def __init__(self):
        print("[QABot] Initializing offline Knowledge Base...")
        self.kb = KnowledgeBase()

        print("[QABot] Initializing Anti-Bot Search & Scraper Engine...")
        self.search = WebSearch()

        print("[QABot] Initializing Overlapping Text Chunker...")
        self.chunker = TextChunker(chunk_size=180, chunk_overlap=40)

        print("[QABot] Loading Semantic Reranker (all-MiniLM-L6-v2)...")
        self.reranker = SemanticReranker(model_name="all-MiniLM-L6-v2")

        print("[QABot] Loading Seq2Seq Generator (facebook/bart-large-cnn)...")
        self.generator = TransformerGenerator(model_name="facebook/bart-large-cnn")

        self.history = []
        print("[QABot] Pipeline ready!")

    def ask(self, query: str, return_details: bool = False) -> Union[str, Dict[str, Any]]:
        """
        Executes the full RAG pipeline:
        1. KB Lookup
        2. Web Search & Safe Scraping (Anti-bot filter + search snippet fallback)
        3. Overlapping Sliding-Window Chunking
        4. Semantic Reranking via Dense Embeddings
        5. Structured JSON Context Formatting
        6. Seq2Seq Transformer Generation (~1500 char rich answer)
        """
        # Step 1: Knowledge Base Lookup
        kb_answer = self.kb.get_answer(query)
        if kb_answer:
            self.history.append((query, kb_answer))
            if return_details:
                return {
                    "answer": kb_answer,
                    "source": "Offline Knowledge Base",
                    "pipeline_meta": {"stage": "knowledge_base", "match": True}
                }
            return kb_answer

        # Step 2: Web Search + Safe Scraper
        documents = self.search.get_clean_documents(query, max_results=4)
        if not documents:
            msg = "Sorry, I couldn't find relevant search results for your query."
            if return_details:
                return {"answer": msg, "source": "None", "pipeline_meta": {"stage": "search", "documents": []}}
            return msg

        # Step 3: Overlapping Sliding-Window Chunking
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk_document(
                text=doc["content"],
                metadata={
                    "url": doc["url"],
                    "title": doc["title"],
                    "source_type": doc["source_type"]
                }
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            msg = "Sorry, could not extract readable text passages from the search results."
            if return_details:
                return {"answer": msg, "source": "None", "pipeline_meta": {"stage": "chunking", "chunks": []}}
            return msg

        # Step 4: Semantic Reranking with SentenceTransformer Bi-Encoder
        top_chunks = self.reranker.rerank_chunks(
            query=query,
            chunks=all_chunks,
            top_k=3,
            similarity_threshold=0.12
        )

        if not top_chunks:
            # Fallback to top chunks even if similarity was moderate
            top_chunks = all_chunks[:2]

        # Step 5 & 6: Structured JSON Context Formatting & Seq2Seq Generation
        gen_result = self.generator.generate_comprehensive_answer(
            query=query,
            top_chunks=top_chunks,
            max_total_chars=1500
        )

        final_answer = gen_result["answer"]
        self.history.append((query, final_answer))

        if return_details:
            return {
                "answer": final_answer,
                "summary": gen_result.get("summary", ""),
                "json_context": gen_result.get("json_context", "{}"),
                "pipeline_meta": {
                    "stage": "generation",
                    "documents": documents,
                    "total_chunks": len(all_chunks),
                    "top_chunks": top_chunks
                }
            }

        return final_answer

    def show_history(self):
        return self.history
