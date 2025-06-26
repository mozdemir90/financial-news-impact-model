"""
News Fetcher Module
Fetches news from various sources using APIs and web scraping
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsFetcher:
    def __init__(self):
        self.news_api_key = config.NEWS_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_newsapi_articles(self, 
                              keywords: List[str], 
                              days_back: int = 30,
                              language: str = 'en') -> List[Dict]:
        """
        Fetch articles from NewsAPI
        
        Args:
            keywords: List of keywords to search for
            days_back: Number of days to look back
            language: Language code ('en' or 'tr')
        
        Returns:
            List of article dictionaries
        """
        articles = []
        base_url = "https://newsapi.org/v2/everything"
        
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        for keyword in keywords:
            params = {
                'q': keyword,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': language,
                'apiKey': self.news_api_key,
                'pageSize': 100
            }
            
            try:
                response = self.session.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data['status'] == 'ok':
                    for article in data['articles']:
                        articles.append({
                            'title': article.get('title', ''),
                            'content': article.get('content', ''),
                            'description': article.get('description', ''),
                            'url': article.get('url', ''),
                            'published_at': article.get('publishedAt', ''),
                            'source': article.get('source', {}).get('name', ''),
                            'language': language,
                            'keyword': keyword
                        })
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching articles for keyword '{keyword}': {e}")
        
        return articles
    
    def scrape_investing_com(self) -> List[Dict]:
        """
        Scrape news from Investing.com
        
        Returns:
            List of article dictionaries
        """
        articles = []
        url = "https://www.investing.com/news/economy"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find news articles (structure may vary)
            news_items = soup.find_all('article', class_='js-article-item')
            
            for item in news_items[:20]:  # Limit to 20 articles
                try:
                    title_elem = item.find('a', class_='title')
                    title = title_elem.text.strip() if title_elem else ''
                    
                    link = title_elem['href'] if title_elem else ''
                    if link and not link.startswith('http'):
                        link = 'https://www.investing.com' + link
                    
                    # Get article content
                    content = self.get_article_content(link) if link else ''
                    
                    articles.append({
                        'title': title,
                        'content': content,
                        'description': '',
                        'url': link,
                        'published_at': datetime.now().isoformat(),
                        'source': 'Investing.com',
                        'language': 'en',
                        'keyword': 'scraped'
                    })
                    
                    time.sleep(0.5)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error scraping article: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error scraping Investing.com: {e}")
        
        return articles
    
    def get_article_content(self, url: str) -> str:
        """
        Extract full content from article URL
        
        Args:
            url: Article URL
            
        Returns:
            Article content text
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Common content selectors
            content_selectors = [
                'div.articlePage',
                'div.article-content',
                'div.content',
                'article',
                'div.post-content'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove script and style elements
                    for script in content_elem(["script", "style"]):
                        script.decompose()
                    return content_elem.get_text(strip=True)
            
            # Fallback: get all paragraph text
            paragraphs = soup.find_all('p')
            return ' '.join([p.get_text(strip=True) for p in paragraphs])
            
        except Exception as e:
            logger.error(f"Error getting content from {url}: {e}")
            return ""
    
    def fetch_all_news(self, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch news from all sources
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            DataFrame with all news articles
        """
        all_articles = []
        
        # Fetch from NewsAPI
        logger.info("Fetching English news from NewsAPI...")
        en_articles = self.fetch_newsapi_articles(
            config.SEARCH_KEYWORDS, days_back, 'en'
        )
        all_articles.extend(en_articles)
        
        logger.info("Fetching Turkish news from NewsAPI...")
        tr_articles = self.fetch_newsapi_articles(
            config.SEARCH_KEYWORDS, days_back, 'tr'
        )
        all_articles.extend(tr_articles)
        
        # Scrape from Investing.com
        logger.info("Scraping from Investing.com...")
        investing_articles = self.scrape_investing_com()
        all_articles.extend(investing_articles)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_articles)
        
        if not df.empty:
            # Clean and process timestamps
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
            df = df.dropna(subset=['published_at'])
            df = df.sort_values('published_at', ascending=False)
            df = df.drop_duplicates(subset=['title'], keep='first')
            
            logger.info(f"Fetched {len(df)} unique articles")
        
        return df

def main():
    """Test the news fetcher"""
    fetcher = NewsFetcher()
    news_df = fetcher.fetch_all_news(days_back=7)
    
    if not news_df.empty:
        news_df.to_csv(config.NEWS_DATA_FILE, index=False)
        print(f"Saved {len(news_df)} articles to {config.NEWS_DATA_FILE}")
        print(news_df.head())
    else:
        print("No articles fetched")

if __name__ == "__main__":
    main()