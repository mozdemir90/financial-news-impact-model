# Finansal Haber Etki Tahmin Modeli

Bu proje, finansal haber metinlerinin Dolar/TRY, Altın ve BIST 100 endeksi üzerindeki potansiyel etkilerini (1-5 arası bir skorla) tahmin etmek için Doğal Dil İşleme (NLP) ve Makine Öğrenmesi modellerini kullanır.

## Özellikler

- **Veri Kaynağı:** NewsAPI
- **Metin Vektörleştirme:** GloVe (glove.6B.100d)
- **Modeller:** RandomForest, XGBoost, LightGBM
- **Arayüz:** Flask ile oluşturulmuş web arayüzü

## Kurulum ve Çalıştırma

1.  Bu depoyu klonlayın:
    ```bash
    git clone https://github.com/mozdemir90/financial-news-impact-model.git
    cd proje-adiniz
    ```

2.  `glove.6B.100d.txt` dosyasını [Stanford GloVe sayfasından](https://nlp.stanford.edu/projects/glove/) indirin ve proje ana dizinine kopyalayın.

3.  Bir sanal ortam oluşturun ve bağımlılıkları yükleyin:
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux için
    # venv\Scripts\activate  # Windows için
    pip install -r requirements.txt
    ```
    
4.  NLTK 'punkt' paketini indirin:
    ```python
    import nltk
    nltk.download('punkt')
    ```

5.  Flask uygulamasını çalıştırın:
    ```bash
    python app.py
    ```

6.  Tarayıcınızda `http://127.0.0.1:5000` adresini açın.
