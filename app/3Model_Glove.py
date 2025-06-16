import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
import joblib # Modelleri kaydetmek için

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


# --- Veri Hazırlama Kısmı (Önceki kodla aynı) ---
# ... (Veri yükleme, temizleme, GloVe ile vektörleştirme ve train/test split kodları buraya gelecek)
# Bu kısmı bir önceki cevaptan kopyalayıp buraya yapıştırabilirsiniz. 
# Önemli olan X_train, X_test, Y_train, Y_test değişkenlerinin hazır olmasıdır.
# Aşağıya sadece model eğitimi ve değerlendirme kısmını ekliyorum.

# --- ÖRNEK VERİ HAZIRLAMA (YUKARIDAKİ ADIMLARI YAPTIĞINIZI VARSAYARAK) ---
print("Veri hazırlanıyor (Bu kısım normalde daha uzun sürer)...")
df = pd.read_csv('data_rated_from_chatgpt.csv')
df.dropna(subset=['summary'], inplace=True)
def clean_text(text): text = str(text).lower(); return re.sub(r'[^a-z\s]', '', text)
df['cleaned_text'] = df['summary'].apply(clean_text)
# Gerçek GloVe yükleme yerine sahte vektörler oluşturalım (hızlı olması için)
# Gerçek kodunuzda GloVe yükleme kısmını kullanın!
df['vector'] = [np.random.rand(100) for _ in range(len(df))]
X = np.stack(df['vector'].values)
Y = df[['dolar_puan', 'altin_puan', 'borsa_puan']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# --- ÖRNEK VERİ HAZIRLAMA SONU ---


# --- 2. Farklı Modelleri Deneme, Değerlendirme ve Kaydetme ---

# Denenecek modelleri bir sözlükte tanımla
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    
    # --- BURASI DEĞİŞTİ ---
    # LightGBM'i MultiOutputRegressor ile sarmala
    "LightGBM": MultiOutputRegressor(LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1))
}

# Her bir modeli eğit, değerlendir ve kaydet
for name, model in models.items():
    print(f"\n{'='*20} {name} Modeli {'='*20}")
    
    print(f"Eğitiliyor...")
    model.fit(X_train, Y_train)
    
    print("Değerlendiriliyor...")
    Y_pred = model.predict(X_test)
    
    target_names = Y.columns
    for i, target_name in enumerate(target_names):
        mae = mean_absolute_error(Y_test.iloc[:, i], Y_pred[:, i])
        r2 = r2_score(Y_test.iloc[:, i], Y_pred[:, i])
        print(f"  -> {target_name.upper()} | MAE: {mae:.3f}, R²: {r2:.3f}")
        
    # Modeli dosyaya kaydet
    model_filename = f"{name}_model.joblib"
    joblib.dump(model, model_filename)
    print(f"\nModel başarıyla '{model_filename}' dosyasına kaydedildi.")

# Ayrıca, bu modellerle kullanılacak olan GloVe modelini ve metin temizleme fonksiyonunu da
# kaydetmek iyi bir fikirdir, ancak şimdilik model dosyaları yeterli.