"""
Web Search & Anti-Bot Resilient Scraper Module
---------------------------------------------
1. Fetches organic search results via DuckDuckGo (title, link, snippet).
2. Uses browser-grade headers with requests.Session.
3. Strips HTML boilerplate and detects paywalls, bot challenges, and login gates.
4. Falls back gracefully to DuckDuckGo search snippets when scraping is blocked.
"""

from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import re

# Try importing ddgs first, fallback to duckduckgo_search
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


# Common bot-block, login-wall, and paywall phrases
BLOCKED_PATTERNS = [
    r"please sign in",
    r"sign in to continue",
    r"sign in to cnn",
    r"log in to your account",
    r"create a free account",
    r"subscribe to continue",
    r"subscribe now",
    r"subscription required",
    r"access denied",
    r"robot check",
    r"not a robot",
    r"verify you are human",
    r"security check",
    r"cloudflare",
    r"enable javascript and cookies",
    r"please enable js",
    r"scan the qr code to download",
    r"turn on javascript",
    r"cookies are disabled",
]
BLOCKED_REGEX = re.compile("|".join(BLOCKED_PATTERNS), re.IGNORECASE)


class WebSearch:
    def __init__(self, timeout: int = 7):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Sec-Ch-Ua": '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Upgrade-Insecure-Requests": "1"
        })

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Fetches top search results from DuckDuckGo including titles, URLs, and snippets.
        """
        results = []
        try:
            with DDGS() as ddgs:
                search_data = ddgs.text(query, max_results=max_results)
                for item in search_data:
                    if isinstance(item, dict) and "href" in item:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("href", ""),
                            "snippet": item.get("body", "")
                        })
        except Exception as e:
            print(f"[WebSearch] DuckDuckGo search error: {e}")
        return results

    def _is_blocked_or_junk(self, text: str) -> bool:
        """
        Detects if content is a paywall, login gate, captcha, or low-information junk.
        """
        if not text or len(text.strip()) < 100:
            return True
        if BLOCKED_REGEX.search(text):
            return True
        return False

    def scrape_url(self, url: str) -> str:
        """
        Scrapes and extracts meaningful text from a webpage, cleaning boilerplate.
        Returns cleaned text if successful and legitimate, or None if blocked/error.
        """
        try:
            resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            if resp.status_code != 200:
                return None

            # Verify it's an HTML page
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            # Remove noise elements
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button", "noscript", "svg", "iframe"]):
                tag.decompose()

            # Attempt to extract from article or main body first
            article_node = soup.find("article") or soup.find("main") or soup.find("div", {"role": "main"})
            search_scope = article_node if article_node else soup

            paragraphs = []
            for p in search_scope.find_all("p"):
                p_text = p.get_text().strip()
                # Exclude one-liner navigational junk or social share buttons
                if len(p_text) >= 40:
                    paragraphs.append(p_text)

            full_text = " ".join(paragraphs)
            clean_text = re.sub(r"\s+", " ", full_text).strip()

            if self._is_blocked_or_junk(clean_text):
                return None

            return clean_text
        except Exception:
            return None

    def get_clean_documents(self, query: str, max_results: int = 4) -> List[Dict[str, Any]]:
        """
        Searches web, attempts scraping, and falls back to snippets when blocked.
        Returns a list of structured documents with source attribution and status.
        """
        search_results = self.search(query, max_results=max_results)
        documents = []

        for item in search_results:
            url = item["url"]
            title = item["title"]
            snippet = item["snippet"]

            scraped_content = self.scrape_url(url)

            if scraped_content and len(scraped_content.split()) >= 60:
                documents.append({
                    "title": title,
                    "url": url,
                    "content": scraped_content,
                    "source_type": "full_article_scraped",
                    "status": "Scraped successfully"
                })
            elif snippet and len(snippet.strip()) >= 30:
                # Graceful fallback: Search snippet ensures immunity to anti-scraping
                documents.append({
                    "title": title,
                    "url": url,
                    "content": snippet,
                    "source_type": "search_snippet_fallback",
                    "status": "Fallback to search snippet (Paywalled/Blocked)"
                })

        return documents
