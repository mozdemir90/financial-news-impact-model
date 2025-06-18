# app.py (Güncellenmiş Hali)

import joblib
import numpy as np
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from flask import Flask, request, render_template
import logging # YENİ: Loglama kütüphanesini import et

# --- 1. Flask Uygulamasını Başlatma ve Loglamayı Ayarlama ---
app = Flask(__name__)

# YENİ: Profesyonel loglama ayarları
# Sadece dosyaya loglama yapacağız. İsterseniz konsola da ekleyebilirsiniz.
file_handler = logging.FileHandler('app.log') # Logların kaydedileceği dosya
file_handler.setLevel(logging.INFO) # Kaydedilecek minimum log seviyesi
# Log formatı: Tarih/Saat - Log Seviyesi - Mesaj
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)


# --- 2. Gerekli Dosyaları ve Modelleri ÖNCEDEN Yükleme ---
app.logger.info("Uygulama başlatılıyor...") # DEĞİŞTİ: print yerine logger kullanıldı

GLOVE_FILE = 'glove.6B.100d.txt'
RF_MODEL_PATH = 'RandomForest_model.joblib'
XGB_MODEL_PATH = 'XGBoost_model.joblib'
LGBM_MODEL_PATH = 'LightGBM_model.joblib'
EMBEDDING_DIM = 100

def load_resources():
    """Tüm gerekli modelleri ve GloVe vektörlerini belleğe yükler."""
    app.logger.info("Kaynaklar yükleniyor...") # DEĞİŞTİ
    
    # GloVe vektörlerini yükle
    embeddings_index = {}
    with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    app.logger.info(f"{len(embeddings_index)} kelimelik GloVe vektörü yüklendi.") # DEĞİŞTİ
    
    # Makine öğrenmesi modellerini yükle
    models = {
        "RandomForest": joblib.load(RF_MODEL_PATH),
        "XGBoost": joblib.load(XGB_MODEL_PATH),
        "LightGBM": joblib.load(LGBM_MODEL_PATH)
    }
    app.logger.info("Tüm ML modelleri başarıyla yüklendi.") # DEĞİŞTİ
    
    return embeddings_index, models

# Kaynakları global değişkenlere ata
glove_vectors, ml_models = load_resources()
app.logger.info("Uygulama artık istekleri kabul etmeye hazır.") # DEĞİŞTİ

# --- 3. Tahmin için Yardımcı Fonksiyonlar (Aynı kaldı) ---

def text_to_vector(text, embeddings_index, embedding_dim):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    
    embedding_matrix = np.zeros((len(words), embedding_dim))
    words_in_vocab = 0
    for i, word in enumerate(words):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            words_in_vocab += 1
            
    if words_in_vocab == 0:
        return np.zeros((embedding_dim,))
    
    text_vector = np.mean(embedding_matrix[embedding_matrix.any(axis=1)], axis=0)
    return text_vector

def predict_impact(text_input):
    vector = text_to_vector(text_input, glove_vectors, EMBEDDING_DIM)
    vector = vector.reshape(1, -1)
    feature_names = [f'feature_{i}' for i in range(EMBEDDING_DIM)]
    vector_df = pd.DataFrame(vector, columns=feature_names)
    
    predictions = {}
    for name, model in ml_models.items():
        pred = model.predict(vector_df)[0]
        predictions[name] = {
            "Dolar": f"{pred[0]:.2f}",
            "Altın": f"{pred[1]:.2f}",
            "Borsa": f"{pred[2]:.2f}"
        }
    app.logger.info(f"Tahmin sonuçları: {predictions}") # YENİ: Tahmin sonucunu logla
    return predictions

# --- 4. Web Sayfası Rotalarını Tanımlama ---

# app.py (Sadece bu fonksiyonu güncelleyin)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    news_text = "" # YENİ: Değişkeni başlangıçta boş bir string olarak tanımla

    # Eğer kullanıcı formu doldurup POST isteği gönderdiyse:
    if request.method == 'POST':
        news_text = request.form['news_text'] # Kullanıcının girdiği metni al
        app.logger.info(f"POST isteği alındı. Gelen metin: '{news_text[:100]}...'")
        if news_text:
            # Tahmin fonksiyonunu çağır
            results = predict_impact(news_text)
    
    # DEĞİŞTİ: Artık 'submitted_text' değişkenini de şablona gönderiyoruz.
    return render_template('index.html', results=results, submitted_text=news_text)

# --- 5. Uygulamayı Çalıştırma ---
if __name__ == '__main__':
    app.run(debug=True)