import pandas as pd
import numpy as np
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from nltk.tokenize import word_tokenize
import nltk

# NLTK 'punkt' paketinin indirilmiş olduğundan emin olun
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- 1. GERÇEK VERİ HAZIRLAMA ---

# GloVe dosyasının yolu ve vektör boyutu
GLOVE_FILE = 'glove.6B.100d.txt' # GloVe dosyanızın yolu
EMBEDDING_DIM = 100

def load_glove_model(glove_file):
    """GloVe kelime vektörlerini bir sözlüğe yükler."""
    print("GloVe modeli yükleniyor... Bu işlem biraz zaman alabilir.")
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"{len(embeddings_index)} kelime vektörü yüklendi.")
    return embeddings_index

def text_to_vector(text, embeddings_index, embedding_dim):
    """Bir metni, kelime vektörlerinin ortalamasını alarak tek bir vektöre dönüştürür."""
    # Metin temizleme adımları
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Sadece harf ve boşluk bırak
    words = word_tokenize(text)
    
    # Kelimelerin vektörlerini bul ve ortalamasını al
    vectors = [embeddings_index.get(word) for word in words if embeddings_index.get(word) is not None]
    
    if not vectors:
        return np.zeros(embedding_dim)
    return np.mean(vectors, axis=0)

# Veriyi ve GloVe modelini yükle
print("Veri ve GloVe modeli yükleniyor...")
df = pd.read_csv('data_rated_from_chatgpt.csv')
df.dropna(subset=['summary'], inplace=True)
glove_model = load_glove_model(GLOVE_FILE)

# Her bir haber metnini vektöre dönüştür
print("Metinler vektörlere dönüştürülüyor...")
df['vector'] = df['summary'].apply(lambda text: text_to_vector(text, glove_model, EMBEDDING_DIM))

# Özellik (X) ve hedef (Y) değişkenlerini oluştur
X = np.stack(df['vector'].values)
Y = df[['dolar_puan', 'altin_puan', 'borsa_puan']]

# Eğitim ve test setlerine ayır
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Veri hazırlama tamamlandı.")

# --- 2. Farklı Modelleri Deneme, Değerlendirme ve Kaydetme ---
# ... BU KISIM SİZİN KODUNUZLA AYNI VE DOĞRU ...
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "LightGBM": MultiOutputRegressor(LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1))
}

for name, model in models.items():
    print(f"\n{'='*20} {name} Modeli {'='*20}")
    
    print(f"Eğitiliyor...")
    # Modeller artık DataFrame feature isimleri görmeyecek, bu yüzden uyarı almayacağız
    if name in ["XGBoost", "LightGBM"]:
         # Bazı modeller uyarı verebilir, uyarıları görmezden gelmek için
         # X_train'i ve X_test'i Pandas DataFrame'e çevirebiliriz ama şimdilik gerekli değil.
         model.fit(X_train, Y_train)
    else:
         model.fit(X_train, Y_train)

    print("Değerlendiriliyor...")
    Y_pred = model.predict(X_test)
    
    target_names = Y.columns
    for i, target_name in enumerate(target_names):
        mae = mean_absolute_error(Y_test.iloc[:, i], Y_pred[:, i])
        r2 = r2_score(Y_test.iloc[:, i], Y_pred[:, i])
        print(f"  -> {target_name.upper()} | MAE: {mae:.3f}, R²: {r2:.3f}")
        
    model_filename = f"{name}_model.joblib"
    joblib.dump(model, model_filename)
    print(f"\nModel başarıyla '{model_filename}' dosyasına kaydedildi.")