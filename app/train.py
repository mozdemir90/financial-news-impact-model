import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import string
from nltk.corpus import stopwords
import nltk

# İndirmenin ilk defa kurulması gerekebilir
nltk.download('stopwords')

# 1. Adım: CSV dosyasını yükle
data = pd.read_csv('Data/cnn_data_labeled.csv')

# Fonksiyon: Metni temizleme
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# 2. Adım: Metinleri temizle
data['cleaned_text'] = data['summary'].apply(clean_text)

# Özellik çıkarımı: TF-IDF
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(data['cleaned_text']).toarray()

# Etiketleri sayısallaştırma
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['label'])

# Eğitim ve test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model seçimi ve eğitimi: Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Tahminler
y_pred = model.predict(X_test)

# Performans metrikleri
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)