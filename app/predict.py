import joblib
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize

# NLTK'nın tokenizer'ı için 'punkt' paketini indirmeniz gerekebilir
# nltk.download('punkt')


nltk.data.path.append("/Users/mozdemir/nltk_data")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# --- 1. Gerekli Dosyaları ve Modelleri Yükleme ---

# Dosya yollarını burada tanımlayın
GLOVE_FILE = 'glove.6B.100d.txt'  # Kullandığınız GloVe dosyasının yolu
RF_MODEL_PATH = 'RandomForest_model.joblib'
XGB_MODEL_PATH = 'XGBoost_model.joblib'
LGBM_MODEL_PATH = 'LightGBM_model.joblib'
EMBEDDING_DIM = 100 # GloVe vektör boyutunuz

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

def load_models():
    """Kaydedilmiş makine öğrenmesi modellerini yükler."""
    print("Kaydedilmiş modeller yükleniyor...")
    models = {
        "RandomForest": joblib.load(RF_MODEL_PATH),
        "XGBoost": joblib.load(XGB_MODEL_PATH),
        "LightGBM": joblib.load(LGBM_MODEL_PATH)
    }
    print("Modeller başarıyla yüklendi.")
    return models

# --- 2. Tahmin için Yardımcı Fonksiyonlar ---

def text_to_vector(text, embeddings_index, embedding_dim):
    """Bir metni, kelime vektörlerinin ortalamasını alarak tek bir vektöre dönüştürür."""
    # Metni temizle (eğitimdekiyle aynı adımlar)
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Sayıları kaldır
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    
    words = word_tokenize(text)
    words = text.split()

    
    # Kelime vektörlerini topla
    embedding_matrix = np.zeros((len(words), embedding_dim))
    words_in_vocab = 0
    for i, word in enumerate(words):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            words_in_vocab += 1
            
    # Eğer metindeki hiçbir kelime GloVe sözlüğünde yoksa, sıfır vektörü döndür
    if words_in_vocab == 0:
        return np.zeros((embedding_dim,))
    
    # Vektörlerin ortalamasını alarak metnin tekil vektörünü oluştur
    text_vector = np.mean(embedding_matrix[embedding_matrix.any(axis=1)], axis=0)
    return text_vector

def predict_impact(text_input, models, embeddings_index):
    """
    Verilen bir haber metni için tüm modellerden etki skorlarını tahmin eder.
    """
    # Metni vektöre dönüştür
    vector = text_to_vector(text_input, embeddings_index, EMBEDDING_DIM)
    
    # Modelin beklediği formata getir: (1, n_features)
    vector = vector.reshape(1, -1)
    
    predictions = {}
    for name, model in models.items():
        # Tahmin yap
        pred = model.predict(vector)[0]  # [0] çünkü tek bir tahmin alıyoruz
        predictions[name] = {
            "Dolar": pred[0],
            "Altın": pred[1],
            "Borsa": pred[2]
        }
    return predictions

# --- 3. Gerçek Testleri Çalıştırma ---

if __name__ == "__main__":
    # GloVe ve ML modellerini yükle
    glove_vectors = load_glove_model(GLOVE_FILE)
    ml_models = load_models()
    
    # Test edilecek yeni haber metinleri
    haber_metni_1 = "Gross domestic product contracted by an estimated 0.3% in April. It was the sharpest month-on-month decline since October 2023. The drop was largely due to a contraction in the services sector. But Trump’s trade war also played a part."
    haber_metni_2 = "US and Chinese officials will meet in London on Monday to discuss trade. The announcement comes after Trump and his Chinese counterpart Xi Jinping spoke for 90 minutes on Thursday. The last talks between the Trump administration and their Chinese counterparts, held on May 12 in Geneva, represented a major turning point for the global trade war."
    haber_metni_3 = "Gold, silver and platinum prices have surged this year. Investors have piled into precious metals in search of places to hide from trade war uncertainty. Gold has soared 27.5%, silver is up 24% and platinum has surged a whopping 36%."

    print("\n--- TAHMİN SONUÇLARI ---\n")

    # 1. Haber için tahmin
    print(f"Haber: '{haber_metni_1}'")
    results_1 = predict_impact(haber_metni_1, ml_models, glove_vectors)
    for model_name, preds in results_1.items():
        print(f"  [{model_name}] -> Dolar: {preds['Dolar']:.2f}, Altın: {preds['Altın']:.2f}, Borsa: {preds['Borsa']:.2f}")
    
    print("-" * 20)

    # 2. Haber için tahmin
    print(f"Haber: '{haber_metni_2}'")
    results_2 = predict_impact(haber_metni_2, ml_models, glove_vectors)
    for model_name, preds in results_2.items():
        print(f"  [{model_name}] -> Dolar: {preds['Dolar']:.2f}, Altın: {preds['Altın']:.2f}, Borsa: {preds['Borsa']:.2f}")

    print("-" * 20)

    # 3. Haber için tahmin
    print(f"Haber: '{haber_metni_3}'")
    results_3 = predict_impact(haber_metni_3, ml_models, glove_vectors)
    for model_name, preds in results_3.items():
        print(f"  [{model_name}] -> Dolar: {preds['Dolar']:.2f}, Altın: {preds['Altın']:.2f}, Borsa: {preds['Borsa']:.2f}")