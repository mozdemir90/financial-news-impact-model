
## 10. Example Usage Script
"""
Example usage of the Turkish Market Impact Predictor
"""

import pandas as pd
from data_generator import generate_complete_dataset
from train_model import MarketImpactPredictor as Trainer
from predict import MarketImpactPredictor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    """Run the complete pipeline from data generation to prediction"""
    
    print("🚀 Starting Turkish Market Impact Predictor Pipeline")
    print("=" * 60)
    
    # Step 1: Generate sample data
    print("\n📊 Step 1: Generating sample data...")
    try:
        generate_complete_dataset()
        print("✅ Sample data generated successfully")
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return
    
    # Step 2: Train model
    print("\n🤖 Step 2: Training prediction model...")
    try:
        trainer = Trainer()
        trainer.train_full_pipeline()
        print("✅ Model trained successfully")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return
    
    # Step 3: Test predictions
    print("\n🔮 Step 3: Testing predictions...")
    try:
        predictor = MarketImpactPredictor()
        predictor.load_model()
        
        # Test with sample articles
        test_articles = [
            {
                'title': 'Turkish Central Bank cuts interest rates unexpectedly',
                'content': 'In a surprise move, the Turkish Central Bank has reduced interest rates by 200 basis points, citing improved inflation outlook.',
                'description': 'Unexpected rate cut by Turkish Central Bank',
                'source': 'Reuters',
                'language': 'en'
            },
            {
                'title': 'Turkey signs major trade agreement with European Union',
                'content': 'Turkey and the EU have reached a comprehensive trade agreement that is expected to boost Turkish exports significantly.',
                'description': 'New trade agreement between Turkey and EU',
                'source': 'Bloomberg',
                'language': 'en'
            }
        ]
        
        for i, article in enumerate(test_articles, 1):
            print(f"\n📰 Test Article {i}:")
            print(f"Title: {article['title']}")
            
            predictions = predictor.predict_single_article(**article)
            summary = predictor.get_market_summary(predictions)
            print(summary)
        
        print("\n✅ Predictions completed successfully")
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        return
    
    # Step 4: Launch web interface instructions
    print("\n🌐 Step 4: Launch Web Interface")
    print("To start the interactive web interface, run:")
    print("streamlit run app.py")
    print("\nThen open your browser to: http://localhost:8501")
    
    print("\n🎉 Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_complete_pipeline()