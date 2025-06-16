import pandas as pd
import json

RESULTS_FILE = 'chatgpt_results.txt' # ChatGPT'den kopyaladığınız sonuçlar
ORIGINAL_FILE = 'data_for_rating_raw.csv'
OUTPUT_FILE = 'data_rated_from_chatgpt.csv'

ratings_list = []
with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            # Her satırı JSON olarak ayrıştır
            ratings_list.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            print(f"Hatalı satır atlandı: {line.strip()}")
            # Hata durumunda nötr bir değer ekle
            ratings_list.append({"dolar": 3, "altin": 3, "borsa": 3})

ratings_df = pd.json_normalize(ratings_list)
ratings_df.columns = ['dolar_puan', 'altin_puan', 'borsa_puan']

original_df = pd.read_csv(ORIGINAL_FILE)
final_df = pd.concat([original_df.iloc[:len(ratings_df)], ratings_df], axis=1)

final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"Puanlanmış veri '{OUTPUT_FILE}' dosyasına kaydedildi.")