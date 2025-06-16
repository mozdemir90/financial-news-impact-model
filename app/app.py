# app.py

import joblib
import numpy as np
import re
import pandas as pd  # DataFrame dönüşümü için
from nltk.tokenize import word_tokenize
from flask import Flask, request, render_template

# --- 1. Flask Uygulamasını Başlatma ---
app = Flask(__name__)

# --- 2. Gerekli Dosyaları ve Modelleri ÖNCEDEN Yükleme ---
# Bu kısım çok önemli! Modelleri ve GloVe dosyasını her istekte değil,
# uygulama başlarken SADECE BİR KEZ yüklüyoruz. Bu, performansı ciddi şekilde artırır.

GLOVE_FILE = 'glove.6B.100d.txt'
RF_MODEL_PATH = 'RandomForest_model.joblib'
XGB_MODEL_PATH = 'XGBoost_model.joblib'
LGBM_MODEL_PATH = 'LightGBM_model.joblib'
EMBEDDING_DIM = 100

def load_resources():
    """Tüm gerekli modelleri ve GloVe vektörlerini belleğe yükler."""
    print("Kaynaklar yükleniyor... Lütfen bekleyin.")
    
    # GloVe vektörlerini yükle
    embeddings_index = {}
    with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # Makine öğrenmesi modellerini yükle
    models = {
        "RandomForest": joblib.load(RF_MODEL_PATH),
        "XGBoost": joblib.load(XGB_MODEL_PATH),
        "LightGBM": joblib.load(LGBM_MODEL_PATH)
    }
    
    print("Tüm kaynaklar başarıyla yüklendi.")
    return embeddings_index, models

# Kaynakları global değişkenlere ata
glove_vectors, ml_models = load_resources()

# --- 3. Tahmin için Yardımcı Fonksiyonlar (predict.py'dan alındı) ---

def text_to_vector(text, embeddings_index, embedding_dim):
    """Bir metni, kelime vektörlerinin ortalamasını alarak tek bir vektöre dönüştürür."""
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
    """Verilen bir haber metni için etki skorlarını tahmin eder."""
    vector = text_to_vector(text_input, glove_vectors, EMBEDDING_DIM)
    vector = vector.reshape(1, -1)
    
    # Scikit-learn uyarısını önlemek için DataFrame'e dönüştürelim
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
    return predictions

# --- 4. Web Sayfası Rotalarını Tanımlama ---

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    # Eğer kullanıcı formu doldurup POST isteği gönderdiyse:
    if request.method == 'POST':
        news_text = request.form['news_text']
        if news_text:
            # Tahmin fonksiyonunu çağır
            results = predict_impact(news_text)
    
    # Hem GET isteğinde (sayfa ilk açıldığında) hem de POST isteğinde
    # (form gönderildiğinde) 'index.html' şablonunu render et.
    # POST durumunda 'results' değişkeni dolu gidecek.
    return render_template('index.html', results=results)

# --- 5. Uygulamayı Çalıştırma ---
if __name__ == '__main__':
    # debug=True, kodda değişiklik yaptığınızda sunucunun otomatik yeniden başlamasını sağlar.
    # Production'da (gerçek ortamda) debug=False olmalıdır.
    app.run(debug=True)