# train_and_evaluate.py

import pandas as pd
import numpy as np
import re
import joblib
import time
from nltk.tokenize import word_tokenize
import nltk

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# Denenecek modeller
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR

# --- 1. Gerekli Dosyaları ve Ayarları Yükleme ---
print("Başlatılıyor: Eğitim ve Değerlendirme Script'i")
DATA_PATH = 'data/data_rated_from_chatgpt.csv'
GLOVE_FILE = 'glove.6B.100d.txt' # GloVe dosyanızın yolu
EMBEDDING_DIM = 100
BEST_MODEL_PATH = 'models/best_model.joblib'

# NLTK 'punkt' paketini indir
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- 2. Yardımcı Fonksiyonlar (Veri İşleme) ---
def load_glove_model(glove_file):
    print("GloVe modeli yükleniyor...")
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def text_to_vector(text, embeddings_index, embedding_dim):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = word_tokenize(text)
    vectors = [embeddings_index.get(word) for word in words if embeddings_index.get(word) is not None]
    if not vectors:
        return np.zeros(embedding_dim)
    return np.mean(vectors, axis=0)

# --- 3. Ana İş Akışı ---

# Veriyi yükle
print(f"'{DATA_PATH}' adresinden veriler yükleniyor...")
df = pd.read_csv(DATA_PATH)
df.dropna(subset=['summary'], inplace=True)

# GloVe ve Vektörleştirme
glove_model = load_glove_model(GLOVE_FILE)
print("Metinler vektörlere dönüştürülüyor...")
df['vector'] = df['summary'].apply(lambda text: text_to_vector(text, glove_model, EMBEDDING_DIM))

# Eğitim ve Test setlerini oluştur
X = np.stack(df['vector'].values)
Y = df[['dolar_puan', 'altin_puan', 'borsa_puan']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Veri seti ayrıldı: {len(X_train)} eğitim, {len(X_test)} test.")

# Denenecek modeller listesi
# Not: SVR ve AdaBoost gibi bazı modeller MultiOutputRegressor gerektirir.
models_to_try = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "LightGBM": MultiOutputRegressor(LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    "SVM (SVR)": MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1)),
    "AdaBoost": MultiOutputRegressor(AdaBoostRegressor(n_estimators=50, random_state=42))
}

results = {}
best_score = float('inf') # MAE'yi minimize edeceğimiz için sonsuzdan başlatıyoruz
best_model_name = None
best_model_instance = None

print("\n--- Model Eğitimi ve Değerlendirmesi Başladı ---")
for name, model in models_to_try.items():
    start_time = time.time()
    print(f"\nModel: {name}")
    
    # Modeli eğit
    model.fit(X_train, Y_train)
    
    # Tahmin yap ve değerlendir
    Y_pred = model.predict(X_test)
    total_mae = 0
    
    for i, target_name in enumerate(Y.columns):
        mae = mean_absolute_error(Y_test.iloc[:, i], Y_pred[:, i])
        r2 = r2_score(Y_test.iloc[:, i], Y_pred[:, i])
        total_mae += mae
        print(f"  -> {target_name.upper():<12} | MAE: {mae:.3f}, R²: {r2:.3f}")
        
    avg_mae = total_mae / len(Y.columns)
    elapsed_time = time.time() - start_time
    print(f"  -> ORTALAMA MAE: {avg_mae:.3f} (Süre: {elapsed_time:.2f}s)")
    results[name] = avg_mae
    
    # En iyi modeli takip et
    if avg_mae < best_score:
        best_score = avg_mae
        best_model_name = name
        best_model_instance = model

print("\n--- Değerlendirme Tamamlandı ---")
print(f"En iyi performansı gösteren model: {best_model_name} (Ortalama MAE: {best_score:.3f})")

# En iyi modeli diske kaydet
print(f"En iyi model '{BEST_MODEL_PATH}' olarak kaydediliyor...")
joblib.dump(best_model_instance, BEST_MODEL_PATH)
print("İşlem başarıyla tamamlandı.")