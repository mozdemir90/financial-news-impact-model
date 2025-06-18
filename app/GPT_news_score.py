import pandas as pd

INPUT_FILE = 'data_for_rating_raw.csv'
TEXT_COLUMN = 'summary'
BATCH_SIZE = 20 # ChatGPT'nin karakter limitine takılmamak için 20'li gruplar halinde yapalım

try:
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"HATA: '{INPUT_FILE}' bulunamadı.")
    exit()

# Yapay Zekaya verilecek ana talimat
SYSTEM_PROMPT = """
I will give you a list of numbered news summaries. Your task is to analyze each summary's likely short-term impact on three Turkish assets: USD/TRY (Dolar), Gold (Altın), and BIST 100 (Borsa).

Rating Scale:
- For USD/TRY (Dolar): A positive score means Lira strengthens (dollar falls). 1 = very negative (dollar soars), 3 = neutral, 5 = very positive (dollar drops).
- For Gold (Altın) and Stock Market (Borsa): A positive score means the asset's value increases. 1 = very negative (price falls), 3 = neutral, 5 = very positive (price rises).

For each number, you MUST respond with ONLY a valid JSON object in the format {"dolar": <score>, "altin": <score>, "borsa": <score>}. Do not add any other text or numbering. Start with the response for number 1, then a new line for number 2, and so on.

Here are the summaries:
---
"""

# Veriyi batch'lere böl
for i in range(0, len(df), BATCH_SIZE):
    batch_df = df.iloc[i:i+BATCH_SIZE]
    
    # Kopyalanacak metni oluştur
    full_prompt = SYSTEM_PROMPT
    for index, row in batch_df.iterrows():
        # index+1 yerine i+index_in_batch+1 daha doğru olur
        line_number = i + (index - batch_df.index[0]) + 1
        full_prompt += f"{line_number}. \"{row[TEXT_COLUMN]}\"\n"

    # Her bir batch için ayrı bir prompt dosyası oluştur
    with open(f'prompt_batch_{i//BATCH_SIZE + 1}.txt', 'w', encoding='utf-8') as f:
        f.write(full_prompt)

print(f"Prompt dosyaları ('prompt_batch_*.txt') başarıyla oluşturuldu. Bu dosyaların içeriğini kopyalayıp ChatGPT'ye yapıştırın.")