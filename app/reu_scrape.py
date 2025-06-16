import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# --- KONFİGÜRASYON ---
# Her site için arama URL'si ve HTML etiketleri (CSS selectors) farklıdır.
# Not: Bu selector'lar siteler tasarımlarını değiştirdiğinde güncellenmelidir.
SITE_CONFIGS = {
    'Reuters': {
        'search_url': 'https://www.reuters.com/site-search/?query={query}',
        'article_selector': 'li.search-results__item__2o_qR',
        'title_selector': 'h3 a',
        'link_selector': 'h3 a',
        'summary_selector': 'p.search-results__item__2o_qR', # Reuters'ta özet yok, başlığı tekrar alabiliriz.
        'base_url': 'https://www.reuters.com'
    },
    'AP News': {
        'search_url': 'https://apnews.com/search?q={query}',
        'article_selector': 'div.Card',
        'title_selector': 'h3.Page-headline',
        'link_selector': 'a',
        'summary_selector': 'p.Page-summary-text',
        'base_url': '' # AP News tam link veriyor
    }
}

def scrape_news(query, max_results_per_site=20):
    """
    Verilen sorgu için yapılandırılmış sitelerden haberleri çeker.
    """
    all_articles = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    for site_name, config in SITE_CONFIGS.items():
        print(f"\nScraping '{site_name}' for query: '{query}'...")
        
        # Sorguyu URL'ye ekle
        url = config['search_url'].format(query=query)
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status() # HTTP hatalarını kontrol et
        except requests.exceptions.RequestException as e:
            print(f"Could not fetch data from {site_name}: {e}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.select(config['article_selector'])
        
        count = 0
        for article in articles:
            if count >= max_results_per_site:
                break

            title_element = article.select_one(config['title_selector'])
            link_element = article.select_one(config['link_selector'])
            summary_element = article.select_one(config['summary_selector'])
            
            if title_element and link_element:
                title = title_element.get_text(strip=True)
                link = link_element['href']
                
                # Link tam URL değilse, base_url ile birleştir
                if not link.startswith('http'):
                    link = config['base_url'] + link
                
                summary = summary_element.get_text(strip=True) if summary_element else title
                
                all_articles.append({'source': site_name, 'title': title, 'summary': summary, 'link': link})
                count += 1
        
        print(f"Found {count} articles from {site_name}.")
        time.sleep(1) # Siteye saygılı olmak için istekler arasında bekle

    return all_articles

# --- ANA İŞLEM ---
QUERY = "Trump"
scraped_data = scrape_news(QUERY, max_results_per_site=50) # Her siteden en fazla 50 haber çek

if scraped_data:
    df_scraped = pd.DataFrame(scraped_data)
    OUTPUT_FILE = f'scraped_{QUERY}_news.csv'
    df_scraped.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"\nTotal of {len(df_scraped)} articles saved to '{OUTPUT_FILE}'.")
else:
    print("\nNo articles were found.")