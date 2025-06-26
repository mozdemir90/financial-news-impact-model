"""
Data Generator Module
Generates dummy market data and creates labeled training dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import config
import logging
import re

logger = logging.getLogger(__name__)

class MarketDataGenerator:
    def __init__(self):
        self.base_values = {
            'USD_TRY': 39.74,
            'GAU_TRY': 4283.0,
            'BIST100': 9438.0,
            'BTC_TRY': 4276984.0
        }
    
    def generate_market_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate synthetic market data with realistic patterns
        
        Args:
            start_date: Start date for data generation
            end_date: End date for data generation
            
        Returns:
            DataFrame with market data
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        market_data = []
        
        # Initialize previous values
        prev_values = self.base_values.copy()
        
        for date in dates:
            # Add some volatility and trends
            changes = {
                'USD_TRY': np.random.normal(0, 0.002),  # 0.2% hourly volatility
                'GAU_TRY': np.random.normal(0, 0.015),  # 1.5% hourly volatility
                'BIST100': np.random.normal(0, 0.01),   # 1% hourly volatility
                'BTC_TRY': np.random.normal(0, 0.03)    # 3% hourly volatility
            }
            
            # Add some correlation patterns
            if changes['USD_TRY'] > 0:  # USD strengthening
                changes['GAU_TRY'] += 0.001  # Gold often moves with USD/TRY
                changes['BIST100'] -= 0.005  # BIST often moves opposite to USD/TRY
            
            # Update values
            current_values = {}
            for indicator, prev_val in prev_values.items():
                new_val = prev_val * (1 + changes[indicator])
                current_values[indicator] = max(new_val, prev_val * 0.8)  # Prevent extreme drops
            
            market_data.append({
                'datetime': date,
                **current_values
            })
            
            prev_values = current_values
        
        return pd.DataFrame(market_data)
    
    def calculate_impact_scores(self, news_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate impact scores for news articles based on market movements
        
        Args:
            news_df: DataFrame with news articles
            market_df: DataFrame with market data
            
        Returns:
            DataFrame with news and impact scores
        """
        labeled_data = []
        
        for idx, news_row in news_df.iterrows():
            news_time = pd.to_datetime(news_row['published_at'])
            
            # Find market data before and after news
            before_time = news_time - timedelta(hours=1)
            after_time = news_time + timedelta(hours=4)  # Look 4 hours ahead
            
            before_data = market_df[market_df['datetime'] <= before_time].tail(1)
            after_data = market_df[market_df['datetime'] >= after_time].head(1)
            
            if len(before_data) > 0 and len(after_data) > 0:
                before_values = before_data.iloc[0]
                after_values = after_data.iloc[0]
                
                # Calculate percentage changes and convert to impact scores (-10 to +10)
                impact_scores = {}
                for indicator in config.MARKET_INDICATORS.keys():
                    if indicator in before_values and indicator in after_values:
                        pct_change = (after_values[indicator] - before_values[indicator]) / before_values[indicator]
                        # Scale to -10 to +10 range (cap at 10% change = score 10)
                        impact_score = np.clip(pct_change * 100, -10, 10)
                        impact_scores[f'{indicator}_impact'] = impact_score
                    else:
                        impact_scores[f'{indicator}_impact'] = 0
                
                # Add news features
                labeled_data.append({
                    'title': news_row['title'],
                    'content': news_row.get('content', ''),
                    'description': news_row.get('description', ''),
                    'language': news_row['language'],
                    'source': news_row['source'],
                    'published_at': news_row['published_at'],
                    **impact_scores
                })
        
        return pd.DataFrame(labeled_data)
    
    def add_sentiment_based_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add sentiment-based labels for better training data
        
        Args:
            df: DataFrame with news articles
            
        Returns:
            DataFrame with sentiment-based impact scores
        """
        # Define keywords and their typical impact on Turkish markets
        impact_keywords = {
            'positive': {
                'keywords': ['growth', 'increase', 'investment', 'export', 'tourism', 'recovery', 'agreement', 'boost', 'rise', 'improvement', 'success', 'record', 'achievement', 'positive','ceasefire', 'peace', 'stability', 'cooperation', 'development'],
                'impacts': {
                    'USD_TRY_impact': -2.0,  # Positive news typically strengthens TRY
                    'GAU_TRY_impact': 1.0,   # Gold might rise with positive economic news
                    'BIST100_impact': 3.0,   # BIST typically rises with positive news
                    'BTC_TRY_impact': 1.5    # Crypto might benefit from positive sentiment
                }
            },
            'negative': {
                'keywords': ['crisis', 'decline', 'fall', 'recession', 'inflation', 'instability', 'conflict', 'sanctions', 'drop', 'loss', 'negative', 'concern', 'risk', 'uncertainty', 'protest', 'strike', 'unrest','war', 'tension','economic downturn'],
                'impacts': {
                    'USD_TRY_impact': 3.0,   # Negative news typically weakens TRY
                    'GAU_TRY_impact': 2.0,   # Gold often rises during uncertainty
                    'BIST100_impact': -4.0,  # BIST typically falls with negative news
                    'BTC_TRY_impact': -1.0   # Crypto might be affected by negative sentiment
                }
            },
            'neutral': {
                'keywords': ['meeting', 'statement', 'report', 'announcement', 'data', 'statistics', 'update', 'review', 'analysis', 'survey', 'forecast', 'projection', 'outlook', 'commentary', 'assessment'],
                'impacts': {
                    'USD_TRY_impact': 0.0,
                    'GAU_TRY_impact': 0.0,
                    'BIST100_impact': 0.0,
                    'BTC_TRY_impact': 0.0
                }
            }
        }
        
        enhanced_data = []
        
        for idx, row in df.iterrows():
            text = (row['title'] + ' ' + str(row['content']) + ' ' + str(row['description'])).lower()
            
            # Calculate sentiment scores
            sentiment_scores = {f'{indicator}_impact': 0.0 for indicator in config.MARKET_INDICATORS.keys()}
            
            for sentiment_type, data in impact_keywords.items():
                keyword_count = sum(1 for keyword in data['keywords'] if keyword in text)
                
                if keyword_count > 0:
                    # Apply impact based on keyword frequency
                    multiplier = min(keyword_count / 3.0, 1.0)  # Cap at 1.0
                    
                    for indicator, base_impact in data['impacts'].items():
                        sentiment_scores[indicator] += base_impact * multiplier
            
            # Add some randomness to make it more realistic
            for indicator in sentiment_scores:
                noise = np.random.normal(0, 0.5)
                sentiment_scores[indicator] += noise
                sentiment_scores[indicator] = np.clip(sentiment_scores[indicator], -10, 10)
            
            enhanced_data.append({
                'title': row['title'],
                'content': row.get('content', ''),
                'description': row.get('description', ''),
                'language': row['language'],
                'source': row['source'],
                'published_at': row['published_at'],
                **sentiment_scores
            })
        
        return pd.DataFrame(enhanced_data)
    
    def generate_sample_news_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate sample news data for testing
        
        Args:
            num_samples: Number of sample news articles to generate
            
        Returns:
            DataFrame with sample news data
        """
        sample_titles = [
            "Turkish Central Bank raises interest rates to combat inflation",
            "Erdogan announces new economic reform package",
            "Turkish lira strengthens against dollar after policy announcement",
            "Istanbul stock exchange hits new record high",
            "Turkish exports reach all-time high in quarterly results",
            "Inflation concerns grow as food prices surge in Turkey",
            "Turkish economy shows signs of recovery in latest GDP data",
            "Central bank governor signals potential rate cuts ahead",
            "Turkish tourism industry bounces back strongly",
            "Geopolitical tensions affect Turkish financial markets",
            "New trade agreement signed between Turkey and EU",
            "Turkish manufacturing sector expands for third consecutive month",
            "Currency volatility increases amid political uncertainty",
            "Turkish banks report strong quarterly earnings",
            "Government announces infrastructure investment program"
        ]
        
        sample_content = [
            "Economic indicators show mixed signals for Turkish market outlook...",
            "Policy makers are considering various options to stabilize the economy...",
            "Market analysts predict continued volatility in the coming weeks...",
            "International investors are closely watching Turkish economic developments...",
            "The government's new economic strategy aims to boost growth and stability..."
        ]
        
        sources = ['Reuters', 'Bloomberg', 'CNN', 'BBC', 'Hurriyet', 'Haberturk', 'Anadolu Agency']
        languages = ['en', 'tr']
        
        sample_data = []
        start_date = datetime.now() - timedelta(days=30)
        
        for i in range(num_samples):
            # Random date within the last 30 days
            random_date = start_date + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            sample_data.append({
                'title': random.choice(sample_titles),
                'content': random.choice(sample_content),
                'description': random.choice(sample_content)[:100] + '...',
                'language': random.choice(languages),
                'source': random.choice(sources),
                'published_at': random_date,
                'url': f'https://example.com/article_{i}',
                'keyword': 'sample'
            })
        
        return pd.DataFrame(sample_data)

def generate_complete_dataset():
    """
    Generate complete dataset with news and market data
    """
    generator = MarketDataGenerator()
    
    # Generate sample news data
    logger.info("Generating sample news data...")
    news_df = generator.generate_sample_news_data(num_samples=500)
    
    # Generate market data
    start_date = datetime.now() - timedelta(days=35)
    end_date = datetime.now()
    
    logger.info("Generating market data...")
    market_df = generator.generate_market_data(start_date, end_date)
    
    # Add sentiment-based labels
    logger.info("Adding sentiment-based labels...")
    labeled_df = generator.add_sentiment_based_labels(news_df)
    
    # Save data
    news_df.to_csv(config.NEWS_DATA_FILE, index=False)
    market_df.to_csv(config.MARKET_DATA_FILE, index=False)
    labeled_df.to_csv(config.PROCESSED_DATA_FILE, index=False)
    
    logger.info(f"Generated {len(news_df)} news articles")
    logger.info(f"Generated {len(market_df)} market data points")
    logger.info(f"Generated {len(labeled_df)} labeled training samples")
    
    return news_df, market_df, labeled_df

def main():
    """Test the data generator"""
    logging.basicConfig(level=logging.INFO)
    generate_complete_dataset()

if __name__ == "__main__":
    main()