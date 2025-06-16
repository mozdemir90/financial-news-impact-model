import pandas as pd
from newsapi import NewsApiClient
import time

# --- KONFİGÜRASYON ---
API_KEY = "09192f26501c437c94e9d1adbb05bdd2"  # <<< KENDİ NEWSAPI ANAHTARINIZI GİRİN
QUERIES = ["Trump", "Federal Reserve OR FED", "inflation"] # Aranacak anahtar kelimeler
FILENAME = 'Data/data_for_rating_raw.csv'

# --- API İstemcisini Başlatma ---
try:
    newsapi = NewsApiClient(api_key=API_KEY)
except Exception as e:
    print(f"API anahtarı ile ilgili bir hata olabilir: {e}")
    exit()

# --- ANA İŞLEM ---
all_processed_articles = []
print("Haberler çekiliyor...")

for query in QUERIES:
    print(f"'{query}' için arama yapılıyor...")
    try:
        articles_data = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='relevancy',
            page_size=100 # Geliştirici planında max 100 sonuç
        )
        
        for article in articles_data['articles']:
            all_processed_articles.append({
                'source': article['source']['name'],
                'title': article['title'],
                'summary': article['description'],
                'link': article['url'],
                'published_at': article['publishedAt']
            })
        print(f"'{query}' için {len(articles_data['articles'])} haber bulundu.")
        time.sleep(1) # API'ye saygılı olmak için kısa bir bekleme

    except Exception as e:
        print(f"'{query}' için arama yapılırken hata oluştu: {e}")
        continue

# Tüm veriyi bir DataFrame'e dönüştür
if not all_processed_articles:
    print("Hiç haber çekilemedi. İşlem sonlandırılıyor.")
    exit()

df = pd.DataFrame(all_processed_articles)
print(f"\nToplam {len(df)} haber çekildi (tekrarlar dahil).")

# Tekrarlanan haberleri başlığa göre temizle
df_unique = df.drop_duplicates(subset=['title']).reset_index(drop=True)
print(f"Tekrarlar temizlendi. Kalan eşsiz haber sayısı: {len(df_unique)}")

# Ham veriyi kaydet
df_unique.to_csv(FILENAME, index=False, encoding='utf-8')
print(f"\nİşlem tamamlandı! Eşsiz haberler '{FILENAME}' dosyasına kaydedildi.")