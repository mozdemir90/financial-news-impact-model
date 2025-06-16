import pandas as pd
from newsapi import NewsApiClient

# --- KONFİGÜRASYON ---
# NewsAPI.org'dan aldığınız API anahtarını buraya yapıştırın
API_KEY = "09192f26501c437c94e9d1adbb05bdd2"  # <<< KENDİ API ANAHTARINIZI GİRİN

# --- API İstemcisini Başlatma ---
try:
    newsapi = NewsApiClient(api_key=API_KEY)
except Exception as e:
    print(f"API anahtarı ile ilgili bir hata olabilir: {e}")
    exit()

# --- ANA İŞLEM ---
QUERY = "FED"
FILENAME = f'api_scraped_{QUERY}_news.csv'

print(f"NewsAPI.org'dan '{QUERY}' ile ilgili haberler çekiliyor...")

try:
    # 'everything' uç noktasını kullanarak geniş bir arama yap
    # q: arama terimi
    # sources: belirli kaynaklar (örn: 'reuters, associated-press')
    # language: dil
    # page_size: sayfa başına sonuç sayısı (max 100)
    all_articles = newsapi.get_everything(
        q=QUERY,
        sources='reuters, associated-press, bbc-news, cnn, fox-news', # Kaynakları artırabiliriz
        language='en',
        sort_by='relevancy', # Alaka düzeyine göre sırala
        page_size=100 # Geliştirici planında maksimum 100 sonuç alınabilir
    )
except Exception as e:
    # API'den gelen yaygın hataları yakala
    if 'apiKeyInvalid' in str(e):
        print("HATA: API anahtarınız geçersiz. Lütfen NewsAPI.org'dan aldığınız anahtarı kontrol edin.")
    elif 'rateLimited' in str(e):
        print("HATA: API istek limitine ulaştınız. Lütfen bir süre sonra tekrar deneyin.")
    else:
        print(f"Beklenmedik bir hata oluştu: {e}")
    exit()


# Veriyi işleyip bir listeye dönüştür
processed_articles = []
for article in all_articles['articles']:
    processed_articles.append({
        'source': article['source']['name'],
        'title': article['title'],
        'summary': article['description'], # API 'description' alanını özet olarak verir
        'link': article['url'],
        'published_at': article['publishedAt']
    })

# Sonuçları kontrol et ve DataFrame'e dönüştür
if processed_articles:
    df = pd.DataFrame(processed_articles)
    df.to_csv(FILENAME, index=False, encoding='utf-8')
    print(f"\nİşlem tamamlandı! Toplam {len(df)} haber '{FILENAME}' dosyasına kaydedildi.")
    print("\nİlk 5 haber:")
    print(df.head())
else:
    print("\nHiç haber bulunamadı. Sorgunuzu veya kaynakları kontrol edin.")