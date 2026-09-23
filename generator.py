"""
Generator Module: Structured Context Formatting & Seq2Seq Generation
---------------------------------------------------------------------
Formats retrieved RAG chunks into structured JSON context representations,
and uses a Seq2Seq transformer (facebook/bart-large-cnn) to synthesize a
comprehensive answer (up to 1500 characters) strictly conditioned on retrieved evidence.
"""

from typing import List, Dict, Any
import json
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class TransformerGenerator:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Loads the Seq2Seq Transformer model and tokenizer.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def format_structured_context(self, query: str, chunks: List[Dict[str, Any]]) -> tuple:
        """
        Converts retrieved chunks into a structured JSON payload for clean prompt representation.
        """
        passages = []
        for i, chunk in enumerate(chunks):
            passages.append({
                "passage_id": i + 1,
                "source_title": chunk.get("source_title", "Unknown"),
                "text": chunk.get("text", "")
            })

        structured_dict = {
            "query": query,
            "passages": passages
        }
        json_repr = json.dumps(structured_dict, indent=2)
        return json_repr, passages

    def generate_comprehensive_answer(
        self,
        query: str,
        top_chunks: List[Dict[str, Any]],
        max_total_chars: int = 1500
    ) -> Dict[str, Any]:
        """
        Generates a detailed summary and formats the final answer with sources.
        """
        if not top_chunks:
            return {
                "answer": "Sorry, I could not find relevant, high-confidence information to answer your query.",
                "json_context": "{}",
                "summary": ""
            }

        json_context, passages = self.format_structured_context(query, top_chunks)

        # Build clean prompt text for BART encoder (up to 1024 tokens)
        combined_text = " ".join([p["text"] for p in passages])
        prompt_text = f"Question: {query}. Information: {combined_text}"

        # Tokenize with truncation to BART limit
        inputs = self.tokenizer(
            [prompt_text],
            max_length=1024,
            return_tensors="pt",
            truncation=True
        ).to(self.device)

        # Generate abstractive summary with expanded token length
        summary_ids = self.model.generate(
            inputs["input_ids"],
            num_beams=4,
            min_length=70,
            max_length=350,
            length_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

        raw_summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

        # Clean any CNN/DailyMail dataset artifacts if present
        cleaned_summary = re.sub(r"(For more information|For confidential support|visit: http).*$", "", raw_summary, flags=re.IGNORECASE).strip()
        if not cleaned_summary:
            cleaned_summary = raw_summary

        # Assemble rich, informative response (aiming for up to 1500 chars)
        answer_parts = [f"### Summary\n{cleaned_summary}\n"]

        # Add key evidence insights from top passage if space permits
        if len(top_chunks) > 0 and len("\n".join(answer_parts)) < 1100:
            top_evidence = top_chunks[0]["text"]
            # Extract 1-2 key sentences from the highest ranked chunk
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_evidence) if len(s.strip()) > 30]
            if sentences:
                evidence_highlight = " ".join(sentences[:2])
                answer_parts.append(f"**Key Finding ({top_chunks[0].get('source_title', 'Top Source')}):**\n> {evidence_highlight}\n")

        # Add source attribution
        sources_seen = set()
        source_links = []
        for c in top_chunks:
            url = c.get("source_url")
            title = c.get("source_title", "Source")
            if url and url not in sources_seen:
                sources_seen.add(url)
                source_links.append(f"- [{title}]({url})")

        if source_links:
            answer_parts.append("**Sources:**\n" + "\n".join(source_links[:3]))

        full_answer = "\n".join(answer_parts)
        if len(full_answer) > max_total_chars:
            full_answer = full_answer[:max_total_chars].rsplit(" ", 1)[0] + "..."

        return {
            "answer": full_answer,
            "json_context": json_context,
            "summary": cleaned_summary
        }
