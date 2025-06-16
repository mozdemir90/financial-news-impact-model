import pandas as pd
import openai
import json
from tqdm import tqdm

# --- KONFİGÜRASYON ---
openai.api_key = "YOUR_OPENAI_API_KEY" # <<< KENDİ OPENAI API ANAHTARINIZI GİRİN

INPUT_FILE = 'data_for_rating_raw.csv' # Faz 1'de oluşturulan dosya
OUTPUT_FILE = 'data_rated_final.csv'   # Puanlanmış nihai dosya
TEXT_COLUMN = 'summary' # Puanlanacak metinlerin olduğu sütun

# Yapay Zekaya verilecek talimat (Prompt)
SYSTEM_PROMPT = """
You are an expert financial analyst specializing in the impact of global news on the Turkish economy.
Your task is to analyze the provided news summary and rate its likely short-term impact on three Turkish assets: USD/TRY exchange rate (Dolar), Gram Gold price (Altın), and the BIST 100 stock index (Borsa).

You must use the following 1-to-5 rating scale:
- For USD/TRY (Dolar): A positive score means the Lira strengthens (dollar falls). 1 = very negative (dollar soars), 3 = neutral, 5 = very positive (dollar drops).
- For Gold (Altın) and Stock Market (Borsa): A positive score means the asset's value increases. 1 = very negative (asset price falls), 3 = neutral, 5 = very positive (asset price rises).

You MUST respond with ONLY a valid JSON object in the format {"dolar": <score>, "altin": <score>, "borsa": <score>}. Do not add any other text.
Example response: {"dolar": 2, "altin": 4, "borsa": 1}
"""

def get_ai_ratings(text):
    """
    Verilen metni OpenAI API'sine gönderir ve 3 varlık için puanları alır.
    """
    if not isinstance(text, str) or not text.strip():
        return {"dolar": 3, "altin": 3, "borsa": 3} # Boş metinler için nötr puan

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this news summary: '{text}'"}
            ],
            temperature=0,
            response_format={"type": "json_object"} # JSON çıktısı istediğimizi belirtiyoruz
        )
        # JSON string'ini bir Python sözlüğüne çevir
        ratings = json.loads(response.choices[0].message.content)
        return ratings
    except Exception as e:
        print(f"Bir hata oluştu: {e}. Metin: '{text[:50]}...'. Nötr puanlar (3) atanıyor.")
        return {"dolar": 3, "altin": 3, "borsa": 3} # Hata durumunda nötr dön

# --- ANA İŞLEM ---
try:
    df = pd.read_csv(INPUT_FILE)
    print(f"'{INPUT_FILE}' dosyası okundu. {len(df)} satır işlenecek.")
except FileNotFoundError:
    print(f"HATA: '{INPUT_FILE}' bulunamadı. Lütfen önce Faz 1 kodunu çalıştırın.")
    exit()

tqdm.pandas(desc="Rating news with AI")

# Her satır için AI'dan puanları al
# Not: Bu işlem API'ye çok sayıda istek göndereceği için zaman alabilir ve maliyetli olabilir.
results = df[TEXT_COLUMN].progress_apply(get_ai_ratings)

# Gelen sonuçları (sözlük listesi) DataFrame'e çevir
ratings_df = pd.json_normalize(results)
ratings_df.columns = ['dolar_puan', 'altin_puan', 'borsa_puan']

# Orijinal DataFrame ile yeni puanları birleştir
final_df = pd.concat([df, ratings_df], axis=1)

# Sonuçları kaydet
final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')

print(f"\nİşlem tamamlandı! Puanlanmış veriler '{OUTPUT_FILE}' dosyasına kaydedildi.")
print("\nOluşturulan verinin ilk 5 satırı:")
print(final_df.head())