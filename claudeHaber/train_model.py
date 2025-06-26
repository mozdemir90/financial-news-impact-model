"""
Model Training Module
Trains machine learning models to predict market impact
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import logging
from typing import Dict, Tuple, Any
import config
from preprocess import TextPreprocessor

logger = logging.getLogger(__name__)

class MarketImpactPredictor:
    def __init__(self):
        self.models = {}
        self.preprocessor = TextPreprocessor()
        self.target_columns = [f'{indicator}_impact' for indicator in config.MARKET_INDICATORS.keys()]
        
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from processed DataFrame
        
        Args:
            df: Processed DataFrame with features and targets
            
        Returns:
            Features (X) and targets (y) arrays
        """
        # Extract features
        X, processed_df = self.preprocessor.extract_features(df, fit_vectorizer=True)
        
        # Extract targets
        y = df[self.target_columns].values
        
        # Handle missing values
        y = np.nan_to_num(y, nan=0.0)
        
        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        return X, y
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create different model configurations
        
        Returns:
            Dictionary of model configurations
        """
        models = {
            'random_forest': MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            ),
            'gradient_boosting': MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            ),
            'ridge': MultiOutputRegressor(
                Ridge(
                    alpha=1.0,
                    random_state=42
                )
            )
        }
        return models
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict]:
        """
        Train multiple models and evaluate their performance
        
        Args:
            X: Feature matrix
            y: Target matrix
            
        Returns:
            Dictionary with model performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        models = self.create_models()
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_metrics = self.calculate_metrics(y_train, y_pred_train)
            test_metrics = self.calculate_metrics(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[model_name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std()
            }
            
            logger.info(f"{model_name} - Test R²: {test_metrics['r2']:.4f}, "
                       f"Test MAE: {test_metrics['mae']:.4f}")
        
        return results
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    def select_best_model(self, results: Dict[str, Dict]) -> Tuple[str, Any]:
        """
        Select the best performing model
        
        Args:
            results: Model training results
            
        Returns:
            Best model name and model object
        """
        best_model_name = max(results.keys(), 
                             key=lambda k: results[k]['test_metrics']['r2'])
        best_model = results[best_model_name]['model']
        
        logger.info(f"Best model: {best_model_name} with R² = {results[best_model_name]['test_metrics']['r2']:.4f}")
        
        return best_model_name, best_model
    
    def save_model(self, model: Any, model_name: str, metrics: Dict):
        """
        Save trained model and metadata
        
        Args:
            model: Trained model object
            model_name: Name of the model
            metrics: Model performance metrics
        """
        model_path = os.path.join(config.MODEL_DIR, f'{model_name}_model.pkl')
        joblib.dump(model, model_path)
        
        # Save preprocessor
        preprocessor_path = os.path.join(config.MODEL_DIR, 'preprocessor.pkl')
        self.preprocessor.save_preprocessor(preprocessor_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'target_columns': self.target_columns,
            'metrics': metrics,
            'feature_count': model.estimators_[0].n_features_in_ if hasattr(model.estimators_[0], 'n_features_in_') else None
        }
        
        metadata_path = os.path.join(config.MODEL_DIR, 'model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Preprocessor saved to {preprocessor_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def train_full_pipeline(self, data_file: str = None):
        """
        Run the complete training pipeline
        
        Args:
            data_file: Path to training data file
        """
        # Load data
        if data_file is None:
            data_file = config.PROCESSED_DATA_FILE
        
        if not os.path.exists(data_file):
            logger.error(f"Data file {data_file} not found. Please run data_generator.py first.")
            return
        
        logger.info(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
        
        # Check if target columns exist
        missing_targets = [col for col in self.target_columns if col not in df.columns]
        if missing_targets:
            logger.error(f"Missing target columns: {missing_targets}")
            return
        
        # Prepare training data
        X, y = self.prepare_training_data(df)
        
        # Train models
        results = self.train_models(X, y)
        
        # Select and save best model
        best_model_name, best_model = self.select_best_model(results)
        self.save_model(best_model, best_model_name, results[best_model_name])
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        for model_name, result in results.items():
            metrics = result['test_metrics']
            print(f"\n{model_name.upper()}:")
            print(f"  R² Score: {metrics['r2']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  CV Score: {result['cv_score_mean']:.4f} ± {result['cv_score_std']:.4f}")
        
        print(f"\nBest Model: {best_model_name}")
        print("="*50)

def main():
    """Main training function"""
    logging.basicConfig(level=logging.INFO)
    
    predictor = MarketImpactPredictor()
    predictor.train_full_pipeline()

if __name__ == "__main__":
    main()