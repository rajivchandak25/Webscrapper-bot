from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------
# Web Search + Summarizer
# -------------------------------
class WebSearch:
    def __init__(self):
        # Load tokenizer and model directly (version-independent)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    def get_top_links(self, query, num_results=5):
        """
        Uses DuckDuckGo search to fetch top links.
        """
        links = []
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=num_results)
                for res in results:
                    if isinstance(res, dict) and "href" in res:
                        links.append(res["href"])
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
        return links

    def fetch_and_summarize(self, url):
        """
        Scrape webpage text and summarize it.
        """
        try:
            response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = " ".join([p.get_text() for p in soup.find_all("p")])

            if len(paragraphs) > 500:
                # Tokenize inputs
                inputs = self.tokenizer([paragraphs[:1000]], max_length=1024, return_tensors="pt", truncation=True)
                
                # Generate summary IDs
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    num_beams=4,
                    min_length=50,
                    max_length=200,
                    early_stopping=True
                )
                
                # Decode summary
                summary = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
                return summary[0]
                
            return paragraphs[:500]
        except Exception as e:
            print(f"Scrape/Summarize error: {e}")
            return None
