import pandas as pd
from transformers import pipeline

# CSV dosyasını yükle
df = pd.read_csv("Data/api_scr.csv")

# Zero-shot sınıflandırma pipeline'ını tanımla
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Etiketler ve hipotez cümlesi
labels = ["Yükseltir", "Düşürür", "Etkisiz"]
hypothesis_template = "Bu haber dolar veya altın fiyatını {}."

# Her özet için sınıflandırma fonksiyonu
def classify_summary(summary):
    try:
        result = classifier(summary, candidate_labels=labels, hypothesis_template=hypothesis_template)
        return result["labels"][0]
    except Exception as e:
        return "Etkisiz"

# Uygula ve yeni sütun ekle
df["etki"] = df["summary"].apply(classify_summary)

# Yeni CSV olarak kaydet
df.to_csv("Data/api_data_labeled.csv", index=False)
print("Etiketlenmiş CSV kaydedildi: api_data_labeled.csv")
