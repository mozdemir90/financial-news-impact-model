"""
Preprocessing Module
Handles text preprocessing, translation, and feature extraction
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Tuple
import logging
from langdetect import detect
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import config
import joblib
import os

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self):
        self.translator = Translator()
        self.stemmer = PorterStemmer()
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        
        # Load stopwords
        try:
            self.stop_words_en = set(stopwords.words('english'))
            self.stop_words_tr = set(stopwords.words('turkish'))
        except:
            self.stop_words_en = set()
            self.stop_words_tr = set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-ZğüşıöçĞÜŞİÖÇ\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def detect_language(self, text: str) -> str:
        """
        Detect text language
        
        Args:
            text: Text string
            
        Returns:
            Language code ('en', 'tr', or 'unknown')
        """
        try:
            if len(text) < 10:
                return 'unknown'
            lang = detect(text)
            return lang if lang in ['en', 'tr'] else 'unknown'
        except:
            return 'unknown'
    
    def translate_text(self, text: str, target_lang: str = 'en') -> str:
        """
        Translate text to target language
        
        Args:
            text: Text to translate
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        try:
            if len(text) < 5:
                return text
            
            detected_lang = self.detect_language(text)
            if detected_lang == target_lang:
                return text
            
            result = self.translator.translate(text, dest=target_lang)
            return result.text if result and result.text else text
            
        except Exception as e:
            logger.warning(f"Translation failed: {e}. Returning original text.")
            return text
    
    def remove_stopwords(self, text: str, language: str = 'en') -> str:
        """
        Remove stopwords from text
        
        Args:
            text: Text string
            language: Language code
            
        Returns:
            Text with stopwords removed
        """
        try:
            tokens = word_tokenize(text)
            stop_words = self.stop_words_en if language == 'en' else self.stop_words_tr
            
            filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
            return ' '.join(filtered_tokens)
        except:
            return text
    
    def stem_text(self, text: str) -> str:
        """
        Apply stemming to text
        
        Args:
            text: Text string
            
        Returns:
            Stemmed text
        """
        try:
            tokens = word_tokenize(text)
            stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
            return ' '.join(stemmed_tokens)
        except:
            return text
    
    def preprocess_text(self, text: str, language: str = 'en') -> str:
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Raw text
            language: Text language
            
        Returns:
            Preprocessed text
        """
        # Clean text
        text = self.clean_text(text)
        
        # Translate to English if not already
        if language != 'en':
            text = self.translate_text(text, 'en')
        
        # Remove stopwords
        text = self.remove_stopwords(text, 'en')
        
        # Apply stemming
        text = self.stem_text(text)
        
        return text
    
    def extract_features(self, df: pd.DataFrame, fit_vectorizer: bool = True) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Extract features from text data
        
        Args:
            df: DataFrame with text columns
            fit_vectorizer: Whether to fit the vectorizer (True for training, False for inference)
            
        Returns:
            Feature matrix and processed DataFrame
        """
        processed_df = df.copy()
        
        # Combine title, content, and description
        processed_df['combined_text'] = (
            processed_df['title'].fillna('') + ' ' +
            processed_df['content'].fillna('') + ' ' +
            processed_df['description'].fillna('')
        )
        
        # Preprocess combined text
        logger.info("Preprocessing text data...")
        processed_df['processed_text'] = processed_df.apply(
            lambda row: self.preprocess_text(row['combined_text'], row.get('language', 'en')),
            axis=1
        )
        
        # Create TF-IDF features
        if fit_vectorizer:
            logger.info("Fitting TF-IDF vectorizer...")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=config.MODEL_PARAMS['max_features'],
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(processed_df['processed_text'])
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted. Please fit first or set fit_vectorizer=True")
            tfidf_features = self.tfidf_vectorizer.transform(processed_df['processed_text'])
        
        # Add additional features
        additional_features = self.extract_additional_features(processed_df)
        
        # Combine TF-IDF with additional features
        tfidf_dense = tfidf_features.toarray()
        all_features = np.hstack([tfidf_dense, additional_features])
        
        # Scale features
        if fit_vectorizer:
            all_features = self.scaler.fit_transform(all_features)
        else:
            all_features = self.scaler.transform(all_features)
        
        return all_features, processed_df
    
    def extract_additional_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract additional numerical features
        
        Args:
            df: DataFrame with processed text
            
        Returns:
            Additional features array
        """
        features = []
        
        for _, row in df.iterrows():
            text = row['processed_text']
            title = row['title']
            
            # Text length features
            text_length = len(text)
            title_length = len(title)
            word_count = len(text.split())
            
            # Sentiment indicators (simple keyword-based)
            positive_words = ['growth', 'increase', 'rise', 'boost', 'positive', 'strong', 'good']
            negative_words = ['decline', 'fall', 'crisis', 'negative', 'weak', 'bad', 'drop']
            
            positive_count = sum(1 for word in positive_words if word in text.lower())
            negative_count = sum(1 for word in negative_words if word in text.lower())
            sentiment_score = positive_count - negative_count
            
            # Source reliability (simple encoding)
            source_reliability = {
                'Reuters': 1.0, 'Bloomberg': 1.0, 'BBC': 0.9,
                'CNN': 0.8, 'Financial Times': 1.0
            }.get(row.get('source', ''), 0.5)
            
            # Language indicator
            is_turkish = 1 if row.get('language') == 'tr' else 0
            
            # Time features (hour of day)
            try:
                pub_time = pd.to_datetime(row['published_at'])
                hour_of_day = pub_time.hour
                is_weekend = pub_time.weekday() >= 5
            except:
                hour_of_day = 12
                is_weekend = False
            
            features.append([
                text_length, title_length, word_count,
                positive_count, negative_count, sentiment_score,
                source_reliability, is_turkish, hour_of_day, int(is_weekend)
            ])
        
        return np.array(features)
    
    def save_preprocessor(self, filepath: str):
        """Save the preprocessor components"""
        preprocessor_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler
        }
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load the preprocessor components"""
        if os.path.exists(filepath):
            preprocessor_data = joblib.load(filepath)
            self.tfidf_vectorizer = preprocessor_data['tfidf_vectorizer']
            self.scaler = preprocessor_data['scaler']
            logger.info(f"Preprocessor loaded from {filepath}")
        else:
            logger.warning(f"Preprocessor file {filepath} not found")

def main():
    """Test the preprocessor"""
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    if os.path.exists(config.PROCESSED_DATA_FILE):
        df = pd.read_csv(config.PROCESSED_DATA_FILE)
        
        preprocessor = TextPreprocessor()
        features, processed_df = preprocessor.extract_features(df)
        
        print(f"Extracted features shape: {features.shape}")
        print(f"Processed {len(processed_df)} articles")
        
        # Save preprocessor
        preprocessor_path = os.path.join(config.MODEL_DIR, 'preprocessor.pkl')
        preprocessor.save_preprocessor(preprocessor_path)
    else:
        print(f"Data file {config.PROCESSED_DATA_FILE} not found. Please run data_generator.py first.")

if __name__ == "__main__":
    main()