import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- KONFİGÜRASYON ---
INPUT_FILE = 'data_rated_from_chatgpt.csv' # Etiketlenmiş nihai dosyanız
GLOVE_FILE = 'glove.6B.100d.txt'         # GloVe dosyanızın yolu
VECTOR_DIM = 100                          # GloVe vektör boyutu

# --- Gerekli NLTK Verisini İndirme ---
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# --- 1. Adım: Veri Yükleme ve Hazırlama ---

print("Veri yükleniyor ve hazırlanıyor...")
try:
    df = pd.read_csv(INPUT_FILE)
    df.dropna(subset=['summary'], inplace=True) # Özet metni boş olan satırları kaldır
except FileNotFoundError:
    print(f"HATA: '{INPUT_FILE}' bulunamadı.")
    exit()

# a. Metin Temizleme Fonksiyonu
english_stopwords = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    meaningful_words = [word for word in words if word not in english_stopwords]
    return " ".join(meaningful_words)

df['cleaned_text'] = df['summary'].apply(clean_text)

# b. GloVe Modelini Yükleme ve Vektör Oluşturma
def load_glove_model(glove_file):
    print("GloVe Modeli yükleniyor...")
    model = {}
    try:
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word = parts[0]
                vector = np.array([float(val) for val in parts[1:]])
                model[word] = vector
        print(f"Model yüklendi. {len(model)} kelime bulundu.")
        return model
    except FileNotFoundError:
        print(f"HATA: GloVe dosyası '{glove_file}' bulunamadı. Lütfen dosya yolunu kontrol edin.")
        exit()

def text_to_vector(text, model, vector_dim):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(vector_dim)
    return np.mean(word_vectors, axis=0)

glove_model = load_glove_model(GLOVE_FILE)
df['vector'] = df['cleaned_text'].apply(lambda text: text_to_vector(text, glove_model, VECTOR_DIM))

# c. Özellik (X) ve Hedefleri (Y) Tanımlama
X = np.stack(df['vector'].values)
Y = df[['dolar_puan', 'altin_puan', 'borsa_puan']]

print(f"\nVeri hazır. X (özellikler) boyutu: {X.shape}, Y (hedefler) boyutu: {Y.shape}")

# d. Veriyi Eğitim ve Test Setlerine Ayırma
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Eğitim seti: {len(X_train)} satır, Test seti: {len(X_test)} satır")

# --- 2. Adım: Modeli Eğitme ---

print("\nRandomForestRegressor modeli eğitiliyor...")
# n_estimators: ağaç sayısı, random_state: tekrarlanabilirlik
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 tüm işlemci çekirdeklerini kullanır
model.fit(X_train, Y_train)
print("Model eğitimi tamamlandı.")

# --- 3. Adım: Modeli Değerlendirme ---

print("\nModel test seti üzerinde değerlendiriliyor...")
Y_pred = model.predict(X_test)

# Her hedef için metrikleri hesapla ve yazdır
target_names = Y.columns
for i, target_name in enumerate(target_names):
    mae = mean_absolute_error(Y_test.iloc[:, i], Y_pred[:, i])
    r2 = r2_score(Y_test.iloc[:, i], Y_pred[:, i])
    
    print(f"\n--- {target_name.upper()} İçin Performans ---")
    print(f"Ortalama Mutlak Hata (MAE): {mae:.3f}")
    print(f"  -> Anlamı: {target_name} tahminlerimiz gerçek değerden ortalama {mae:.3f} puan sapıyor.")
    print(f"R-Kare (R²): {r2:.3f}")
    print(f"  -> Anlamı: Modelimiz, {target_name} puanlarındaki değişimin yaklaşık %{r2*100:.1f}'ini açıklayabiliyor.")

# --- 4. Adım: Yeni Bir Haberle Tahmin Yapma ---

print("\n--- Yeni Haber Tahmini Testi ---")
new_headline_summary = "The US reported surprisingly low inflation figures, leading to speculation that the Federal Reserve will stop raising interest rates."

# 1. Metni temizle ve vektöre dönüştür
cleaned_new_text = clean_text(new_headline_summary)
vector_new_text = text_to_vector(cleaned_new_text, glove_model, VECTOR_DIM).reshape(1, -1)

# 2. Tahmin yap
predicted_scores = model.predict(vector_new_text)

print(f"Haber Özeti: '{new_headline_summary}'")
print("\nModelin Tahmin Ettiği Puanlar:")
print(f"  -> Dolar Puanı: {predicted_scores[0, 0]:.2f} (Yaklaşık: {round(predicted_scores[0, 0])})")
print(f"  -> Altın Puanı: {predicted_scores[0, 1]:.2f} (Yaklaşık: {round(predicted_scores[0, 1])})")
print(f"  -> Borsa Puanı: {predicted_scores[0, 2]:.2f} (Yaklaşık: {round(predicted_scores[0, 2])})")