"""
Prediction Module
Makes predictions on new news articles
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import Dict, List, Union
import config
from preprocess import TextPreprocessor

logger = logging.getLogger(__name__)

class MarketImpactPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.metadata = None
        self.target_columns = [f'{indicator}_impact' for indicator in config.MARKET_INDICATORS.keys()]
        
    def load_model(self, model_dir: str = None):
        """
        Load trained model and preprocessor
        
        Args:
            model_dir: Directory containing model files
        """
        if model_dir is None:
            model_dir = config.MODEL_DIR
        
        # Load metadata
        metadata_path = os.path.join(model_dir, 'model_metadata.pkl')
        if os.path.exists(metadata_path):
            self.metadata = joblib.load(metadata_path)
            model_name = self.metadata['model_name']
            logger.info(f"Loading model: {model_name}")
        else:
            logger.warning("Model metadata not found. Using default model name.")
            model_name = 'random_forest'
        
        # Load model
        model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load preprocessor
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            self.preprocessor.load_preprocessor(preprocessor_path)
            logger.info("Preprocessor loaded successfully")
        else:
            logger.warning("Preprocessor not found. Using default configuration.")
    
    def predict_single_article(self, 
                              title: str, 
                              content: str = "", 
                              description: str = "",
                              source: str = "Unknown",
                              language: str = "en") -> Dict[str, float]:
        """
        Predict market impact for a single article
        
        Args:
            title: Article title
            content: Article content
            description: Article description
            source: News source
            language: Article language
            
        Returns:
            Dictionary with impact predictions for each market indicator
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        # Create DataFrame for preprocessing
        article_df = pd.DataFrame([{
            'title': title,
            'content': content,
            'description': description,
            'source': source,
            'language': language,
            'published_at': pd.Timestamp.now()
        }])
        
        # Extract features
        X, _ = self.preprocessor.extract_features(article_df, fit_vectorizer=False)
        
        # Make prediction
        prediction = self.model.predict(X)[0]  # Get first (and only) prediction
        
        # Create result dictionary
        result = {}
        for i, indicator in enumerate(config.MARKET_INDICATORS.keys()):
            # Clip predictions to -10, +10 range
            impact_score = np.clip(prediction[i], -10, 10)
            result[indicator] = float(impact_score)
        
        return result
    
    def predict_batch(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict market impact for multiple articles
        
        Args:
            articles_df: DataFrame with article data
            
        Returns:
            DataFrame with predictions added
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please call load_model() first.")
        
        # Extract features
        X, processed_df = self.preprocessor.extract_features(articles_df, fit_vectorizer=False)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Add predictions to dataframe
        result_df = processed_df.copy()
        for i, indicator in enumerate(config.MARKET_INDICATORS.keys()):
            # Clip predictions to -10, +10 range
            result_df[f'{indicator}_impact_pred'] = np.clip(predictions[:, i], -10, 10)
        
        return result_df
    
    def interpret_prediction(self, predictions: Dict[str, float]) -> Dict[str, str]:
        """
        Interpret prediction scores as human-readable text
        
        Args:
            predictions: Dictionary with prediction scores
            
        Returns:
            Dictionary with interpretations
        """
        interpretations = {}
        
        for indicator, score in predictions.items():
            if score > 3:
                interpretation = f"Strong positive impact (+{score:.1f})"
            elif score > 1:
                interpretation = f"Moderate positive impact (+{score:.1f})"
            elif score > -1:
                interpretation = f"Minimal impact ({score:.1f})"
            elif score > -3:
                interpretation = f"Moderate negative impact ({score:.1f})"
            else:
                interpretation = f"Strong negative impact ({score:.1f})"
            
            interpretations[indicator] = interpretation
        
        return interpretations
    
    def get_market_summary(self, predictions: Dict[str, float]) -> str:
        """
        Generate overall market impact summary
        
        Args:
            predictions: Dictionary with prediction scores
            
        Returns:
            Summary string
        """
        interpretations = self.interpret_prediction(predictions)
        
        summary = "MARKET IMPACT PREDICTION\n"
        summary += "=" * 30 + "\n"
        
        for indicator, description in config.MARKET_INDICATORS.items():
            if indicator in predictions:
                score = predictions[indicator]
                interpretation = interpretations[indicator]
                summary += f"{description}: {interpretation}\n"
        
        # Overall sentiment
        avg_score = np.mean(list(predictions.values()))
        if avg_score > 1:
            overall = "Overall: POSITIVE market sentiment expected"
        elif avg_score < -1:
            overall = "Overall: NEGATIVE market sentiment expected"
        else:
            overall = "Overall: NEUTRAL market sentiment expected"
        
        summary += "\n" + overall
        return summary

def demo_prediction():
    """
    Demonstration of prediction functionality
    """
    # Sample news articles for testing
    sample_articles = [
        {
            'title': 'Turkish Central Bank raises interest rates to combat inflation',
            'content': 'The Central Bank of Turkey has announced a significant increase in interest rates as part of efforts to control rising inflation. This move is expected to strengthen the Turkish Lira.',
            'description': 'Central bank announces rate hike to fight inflation',
            'source': 'Reuters',
            'language': 'en'
        },
        {
            'title': 'Turkish exports reach record high in quarterly results',
            'content': 'Turkey has reported record-breaking export figures for the quarter, showing strong economic performance in international trade.',
            'description': 'Export figures show strong economic performance',
            'source': 'Bloomberg',
            'language': 'en'
        },
        {
            'title': 'Political tensions rise affecting financial markets',
            'content': 'Increasing political uncertainty has led to volatility in Turkish financial markets, with investors showing caution.',
            'description': 'Political uncertainty affects markets',
            'source': 'CNN',
            'language': 'en'
        }
    ]
    
    try:
        # Initialize predictor and load model
        predictor = MarketImpactPredictor()
        predictor.load_model()
        
        print("TURKISH MARKET IMPACT PREDICTOR")
        print("=" * 40)
        
        for i, article in enumerate(sample_articles, 1):
            print(f"\nARTICLE {i}:")
            print(f"Title: {article['title']}")
            print("-" * 40)
            
            # Make prediction
            predictions = predictor.predict_single_article(**article)
            
            # Get summary
            summary = predictor.get_market_summary(predictions)
            print(summary)
            print("-" * 40)
    
    except Exception as e:
        logger.error(f"Demo prediction failed: {e}")
        print("Error: Could not load model. Please ensure the model is trained first.")
        print("Run: python train_model.py")

def main():
    """Main prediction function"""
    logging.basicConfig(level=logging.INFO)
    demo_prediction()

if __name__ == "__main__":
    main()