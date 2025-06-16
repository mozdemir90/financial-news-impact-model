# cnn_scraper.py
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from transformers import pipeline
from datetime import datetime
import json
import time
import os  # Klasör oluşturmak için os modülünü ekledik

# === YENİ EKLENEN KISIM ===
# Oluşturduğumuz dönüştürücü modülünden fonksiyonu import ediyoruz
from converter_module import json_to_csv
# ==========================

# Summarizer modelini başlat
# Not: İlk çalıştırmada model indirileceği için biraz zaman alabilir.
print("Summarizer modeli yükleniyor...")
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1  # CPU kullanımı için, GPU varsa 0 yapabilirsiniz
)
print("Model yüklendi.")


# CNN'den haber linklerini çeker
def get_cnn_articles(max_articles=5):
    url = "https://edition.cnn.com/business"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Hata durumunda exception fırlat
    except requests.exceptions.RequestException as e:
        print(f"Hata: CNN ana sayfasına ulaşılamadı. {e}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")

    article_links = []
    # CNN'in yapısı değişebileceği için daha spesifik bir arama yapalım
    for container in soup.find_all('div', {'class': 'container__item'}):
        link_tag = container.find('a', {'data-link-type': 'article'})
        if link_tag and 'href' in link_tag.attrs:
            href = link_tag['href']
            if href.startswith('/'):
                full_url = "https://edition.cnn.com" + href
                if full_url not in article_links:
                    article_links.append(full_url)
        if len(article_links) >= max_articles:
            break
            
    return article_links

# Tekil bir makaleyi indirip özetler
def scrape_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()

        text = article.text.strip()
        if len(text) < 200: # Özetlenemeyecek kadar kısa metinleri atla
            return None

        # Metni modele uygun boyuta getir
        summary = summarizer(text[:1024], max_length=150, min_length=40, do_sample=False)[0]['summary_text']
        
        return {
            "source": "cnn",
            "url": url,
            "title": article.title,
            "summary": summary,
            "published": article.publish_date.isoformat() if article.publish_date else datetime.utcnow().isoformat()
        }
    except Exception as e:
        print(f"Makale işlenirken hata oluştu ({url}): {e}")
        return None

# Makaleleri toplar ve JSON dosyasına kaydeder
def collect_and_save_cnn_data(out_dir="Data", max_articles=5):
    # Data klasörü yoksa oluştur
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    json_out_path = os.path.join(out_dir, "cnn_data.json")
    articles = []
    urls = get_cnn_articles(max_articles=max_articles)
    
    if not urls:
        print("Hiç makale linki bulunamadı. İşlem durduruluyor.")
        return

    for url in urls:
        print(f"İşleniyor: {url}")
        data = scrape_article(url)
        if data:
            articles.append(data)
            print(f"✅ Toplandı: {data['title']}")
        else:
            print(f"❌ Atlandı: {url}")
        time.sleep(1.5)  # engellenmemek için bekle

    if not articles:
        print("Hiçbir makale başarıyla işlenemedi.")
        return

    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"\n🎉 {len(articles)} makale {json_out_path} dosyasına kaydedildi.")

    # === YENİ EKLENEN KISIM ===
    # JSON kaydı bittikten sonra CSV'ye dönüştürme işlemini çağırıyoruz
    print("\nCSV formatına dönüştürülüyor...")
    csv_out_path = json_out_path.replace(".json", ".csv")
    
    if json_to_csv(json_out_path, csv_out_path):
        print(f"✅ Başarılı! Veriler {csv_out_path} dosyasına da kaydedildi.")
    else:
        print("❌ CSV dönüştürme sırasında bir hata oluştu.")
    # ==========================


# Dosya doğrudan çalıştırıldığında çalışsın
if __name__ == "__main__":
    # Örnek olarak 3 makale çekelim
    collect_and_save_cnn_data(max_articles=50)