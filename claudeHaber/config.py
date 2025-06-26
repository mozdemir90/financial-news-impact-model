"""
Configuration file for Turkish Market Analyzer
Contains all constants, API keys, and model parameters
"""

import os

# API Keys (set these as environment variables)
NEWS_API_KEY = os.getenv('NEWS_API_KEY', 'your_newsapi_key_here')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', 'your_alpha_vantage_key_here')

# News Sources Configuration
NEWS_SOURCES = {
    'english': [
        'reuters', 'cnn', 'bbc-news', 'financial-times',
        'bloomberg', 'the-wall-street-journal'
    ],
    'turkish': [
        'haberturk', 'hurriyet', 'milliyet'
    ]
}

# Search Keywords for Turkish Market
SEARCH_KEYWORDS = [
    'Turkey', 'Turkish', 'lira', 'Erdogan', 'CBRT', 'Central Bank Turkey',
    'Istanbul', 'BIST', 'Turkish economy', 'inflation Turkey','interest rates Turkey',
    'Turkish exports', 'Turkish imports', 'Turkey trade balance',
    'Turkey GDP', 'Turkey economic growth', 'Turkey financial markets',
    'Turkey political news', 'Turkey economic news', 'Turkey market news',
    'Turkey stock market', 'Turkey currency', 'Turkey gold price',
    'Turkey Bitcoin', 'Turkey crypto', 'Turkey interest rate decision',
    'Turkey economic indicators', 'Turkey market analysis', 'Turkey news',
    'Turkey financial news', 'Turkey investment', 'Turkey business news',
    'Turkey economic outlook', 'Turkey market trends', 'Turkey economic policy'
]

# Market Indicators
MARKET_INDICATORS = {
    'USD_TRY': 'USD/TRY Exchange Rate',
    'GAU_TRY': 'Gold/TRY Price',
    'BIST100': 'BIST 100 Index',
    'BTC_TRY': 'Bitcoin/TRY Price'
}

# Model Parameters
MODEL_PARAMS = {
    'max_features': 10000,
    'max_length': 512,
    'embedding_dim': 100,
    'lstm_units': 64,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50
}

# File Paths
DATA_DIR = 'data'
MODEL_DIR = 'models'
NEWS_DATA_FILE = os.path.join(DATA_DIR, 'news_data.csv')
MARKET_DATA_FILE = os.path.join(DATA_DIR, 'market_data.csv')
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'processed_data.csv')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)