import pandas as pd
import requests
import json
import time
from tqdm import tqdm

# --- KONFİGÜRASYON ---
# 1. ADIM: BU TOKEN'I KESİNLİKLE KENDİ TOKEN'INIZLA DEĞİŞTİRDİĞİNİZDEN EMİN OLUN!
HF_API_TOKEN = "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" # <<< KENDİ HUGGING FACE TOKEN'INIZI GİRİN

# 2. HİÇBİR KOŞUL GEREKTİRMEYEN AÇIK ERİŞİMLİ MODELİ KULLANACAĞIZ
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

INPUT_FILE = 'data_for_rating_raw.csv'
OUTPUT_FILE = 'data_rated_final_hf_zephyr.csv' # Çıktı dosyasının adını değiştirelim
TEXT_COLUMN = 'summary'

# Prompt'ta bir değişiklik yok
SYSTEM_PROMPT = """
[INST] You are an expert financial analyst. Analyze the following news summary's impact on the Turkish economy (USD/TRY, Gold, BIST 100).

Use this rating scale:
- USD/TRY (Dolar): 1 = dollar soars (bad), 3 = neutral, 5 = dollar drops (good).
- Gold (Altın) and Stock Market (Borsa): 1 = price falls (bad), 3 = neutral, 5 = price rises (good).

Respond with ONLY a valid JSON object like {{"dolar": score, "altin": score, "borsa": score}}.

News Summary: "{text}"[/INST]
"""

headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# get_hf_ratings fonksiyonunda değişiklik yok, önceki "konuşkan" versiyonu kullanabiliriz.
def get_hf_ratings(text):
    if not isinstance(text, str) or not text.strip():
        return {"dolar": 3, "altin": 3, "borsa": 3}
    
    prompt = SYSTEM_PROMPT.format(text=text.replace('"', "'"))
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 50, "return_full_text": False}}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 503:
            print("HATA: Model yükleniyor (503). 30 saniye bekleniyor...")
            time.sleep(30)
            return {"dolar": 3, "altin": 3, "borsa": 3}
        if response.status_code != 200:
            print(f"HATA: API'den beklenmedik durum kodu: {response.status_code} - {response.text}")
            return {"dolar": 3, "altin": 3, "borsa": 3}
        
        try:
            result_text = response.json()[0]['generated_text'].strip()
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_part = result_text[json_start:json_end]
                parsed_json = json.loads(json_part)
                if all(key in parsed_json for key in ["dolar", "altin", "borsa"]):
                    return parsed_json
                else:
                    print(f"HATA: JSON'da beklenen anahtarlar eksik. Alınan: {json_part}")
                    return {"dolar": 3, "altin": 3, "borsa": 3}
            else:
                print(f"HATA: Modelin cevabında JSON bulunamadı. Cevap: {result_text}")
                return {"dolar": 3, "altin": 3, "borsa": 3}
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"HATA: Cevap ayrıştırılamadı ({type(e).__name__}). Cevap: {response.text}")
            return {"dolar": 3, "altin": 3, "borsa": 3}
    except requests.exceptions.RequestException as e:
        print(f"HATA: API isteği başarısız oldu ({type(e).__name__}): {e}")
        return {"dolar": 3, "altin": 3, "borsa": 3}

# --- ANA İŞLEM ---
df = pd.read_csv(INPUT_FILE)
tqdm.pandas(desc="Rating news with Hugging Face API")
results = df[TEXT_COLUMN].progress_apply(get_hf_ratings)

ratings_df = pd.json_normalize(results)
for col in ['dolar', 'altin', 'borsa']:
    if col not in ratings_df.columns:
        ratings_df[col] = 3
ratings_df.columns = ['dolar_puan', 'altin_puan', 'borsa_puan']

final_df = pd.concat([df.reset_index(drop=True), ratings_df.reset_index(drop=True)], axis=1)
final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"\nPuanlanmış veri '{OUTPUT_FILE}' dosyasına kaydedildi.")