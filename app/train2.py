import pandas as pd
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords

# --- Konfigurasyon ---
# Çıktıda metinlerin kesilmemesi için
pd.set_option('display.max_colwidth', 150)

# --- NLTK Verisini İndirme ---
try:
    stopwords.words('english')
except LookupError:
    print("Downloading English stopwords list from NLTK...")
    nltk.download('stopwords')
    print("Download complete.")

# --- 1. Adım: Veri Yükleme ---
FILE_PATH = 'Data/api_data_labeled.csv'  # <<< KENDİ DOSYA YOLUNUZLA DEĞİŞTİRİN
try:
    data = pd.read_csv(FILE_PATH)
    print("Dataset loaded successfully.")
    print("Dataset Shape:", data.shape)
except FileNotFoundError:
    print(f"ERROR: CSV file not found at '{FILE_PATH}'.")
    exit()

# --- 2. Adım: Veri Hazırlama ---

# a. Metin Temizleme Fonksiyonu
english_stopwords = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    meaningful_words = [word for word in words if word not in english_stopwords]
    return " ".join(meaningful_words)

print("\nStarting text cleaning process...")
TEXT_COLUMN = 'summary'  # <<< KENDİ SÜTUN ADINIZLA DEĞİŞTİRİN
data['cleaned_text'] = data[TEXT_COLUMN].apply(clean_text)
print(f"Text in '{TEXT_COLUMN}' column has been cleaned.")

# b. Özellik Çıkarımı (TF-IDF)
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['cleaned_text']).toarray()

# c. Etiketleri Sayısallaştırma
LABEL_COLUMN = 'label'  # <<< KENDİ SÜTUN ADINIZLA DEĞİŞTİRİN
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data[LABEL_COLUMN])

# d. Eğitim & Test Ayrımı
# ÖNEMLİ: Orijinal metinleri de ayırıyoruz ki tahminlerle karşılaştırabilelim.
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X, y, data[TEXT_COLUMN], # Orijinal metin sütununu da ayırmaya dahil et
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"\nDataset split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")


# --- 3. Adım: Sınıflandırıcıları Eğitme ve Karşılaştırma ---

# Eğitilecek modelleri tanımla
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(random_state=42)
}

# Her bir modeli döngüde eğit, değerlendir ve test tahminlerini göster
for name, model in models.items():
    print(f"\n{'='*25} {name} {'='*25}")
    
    # Modeli eğit
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    print("Training complete.")
    
    # Test seti üzerinde tahmin yap
    y_pred = model.predict(X_test)
    
    # a. Performans Raporunu Yazdır
    print("\n--- Performance Report ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
    print(report)
    
    # b. Örnek Test Tahminlerini Yazdır
    print(f"\n--- Sample Test Predictions for {name} ---")
    
    # Sayısal etiketleri metin etiketlerine geri çevir
    true_labels_text = label_encoder.inverse_transform(y_test)
    pred_labels_text = label_encoder.inverse_transform(y_pred)
    
    # Sonuçları göstermek için bir DataFrame oluştur
    results_df = pd.DataFrame({
        'Original Text': text_test,
        'True Label': true_labels_text,
        'Predicted Label': pred_labels_text
    })
    
    # Sadece ilk 10 örneği göster
    print(results_df.head(10))

print(f"\n{'='*60}\n")


# --- 4. Adım: Yeni Bir Başlıkla Tahmin Yapma ---
print("\n--- New Headline Prediction Test ---")
new_headline_text = "The Federal Reserve announced a significant interest rate hike to combat inflation."
print(f"Headline: '{new_headline_text}'\n")

cleaned_text = clean_text(new_headline_text)
vectorized_text = vectorizer.transform([cleaned_text])

# Eğitilmiş her modelden bir tahmin al
for name, model in models.items():
    predicted_label_numeric = model.predict(vectorized_text)
    predicted_label = label_encoder.inverse_transform(predicted_label_numeric)
    print(f"Prediction from {name}: {predicted_label[0]}")