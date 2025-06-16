import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

HEADERS = {
    'User-Agent': os.getenv("USER_AGENT")
}

def get_news_from_cnbc():
    url = "https://www.cnbc.com/world/?region=world"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print("CNBC erişim hatası:", response.status_code)
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = soup.find_all("a", class_="Card-title")

    news = []
    for h in headlines:
        text = h.get_text().strip()
        link = h.get('href')
        if text and link:
            news.append((text, link))

    return news

if __name__ == "__main__":
    results = get_news_from_cnbc()
    for title, link in results:
        print(f"- {title}\n  {link}")
