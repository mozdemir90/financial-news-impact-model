"""
Streamlit Web Application
Interactive interface for Turkish Market Impact Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import config
from predict import MarketImpactPredictor
from news_fetcher import NewsFetcher
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Turkish Market Impact Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

def load_model():
    """Load the prediction model"""
    try:
        if st.session_state.predictor is None:
            st.session_state.predictor = MarketImpactPredictor()
        
        st.session_state.predictor.load_model()
        st.session_state.model_loaded = True
        return True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.error("Please ensure the model is trained. Run: python train_model.py")
        return False

def create_impact_chart(predictions):
    """Create a radar chart for impact predictions"""
    indicators = list(config.MARKET_INDICATORS.keys())
    values = [predictions.get(indicator, 0) for indicator in indicators]
    labels = [config.MARKET_INDICATORS[indicator] for indicator in indicators]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
        name='Impact Score',
        line_color='rgb(0, 123, 255)',
        fillcolor='rgba(0, 123, 255, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-10, 10],
                tickvals=[-10, -5, 0, 5, 10],
                ticktext=['-10', '-5', '0', '+5', '+10']
            )
        ),
        showlegend=False,
        title="Market Impact Prediction",
        title_x=0.5
    )
    
    return fig

def create_bar_chart(predictions):
    """Create a bar chart for impact predictions"""
    indicators = list(config.MARKET_INDICATORS.keys())
    values = [predictions.get(indicator, 0) for indicator in indicators]
    labels = [config.MARKET_INDICATORS[indicator] for indicator in indicators]
    
    colors = ['red' if v < 0 else 'green' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(x=labels, y=values, marker_color=colors)
    ])
    
    fig.update_layout(
        title="Market Impact Scores",
        xaxis_title="Market Indicators",
        yaxis_title="Impact Score (-10 to +10)",
        yaxis=dict(range=[-10, 10]),
        showlegend=False
    )
    
    return fig

def main():
    """Main Streamlit application"""
    st.title("🇹🇷 Turkish Market Impact Predictor")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Single Article Prediction",
        "Batch Prediction",
        "Live News Analysis",
        "Model Information"
    ])
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading prediction model..."):
            if not load_model():
                st.stop()
    
    # Page routing
    if page == "Single Article Prediction":
        single_article_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Live News Analysis":
        live_news_page()
    elif page == "Model Information":
        model_info_page()

def single_article_page():
    """Single article prediction page"""
    st.header("📰 Single Article Prediction")
    st.markdown("Enter news article details to predict market impact:")
    
    # Input form
    with st.form("article_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input("Article Title *", 
                                placeholder="Enter the news headline...")
            
            source = st.selectbox("News Source", [
                "Reuters", "Bloomberg", "CNN", "BBC", "Financial Times",
                "Hurriyet", "Haberturk", "Milliyet", "Other"
            ])
            
            language = st.selectbox("Language", ["en", "tr"])
        
        with col2:
            content = st.text_area("Article Content", 
                                 placeholder="Enter the full article content...",
                                 height=100)
            
            description = st.text_area("Article Description", 
                                     placeholder="Enter article summary or description...",
                                     height=100)
        
        submitted = st.form_submit_button("Predict Market Impact")
    
    if submitted and title:
        with st.spinner("Analyzing article and predicting market impact..."):
            try:
                # Make prediction
                predictions = st.session_state.predictor.predict_single_article(
                    title=title,
                    content=content,
                    description=description,
                    source=source,
                    language=language
                )
                
                # Display results
                st.success("Prediction completed!")
                
                # Create two columns for charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Radar chart
                    radar_fig = create_impact_chart(predictions)
                    st.plotly_chart(radar_fig, use_container_width=True)
                
                with col2:
                    # Bar chart
                    bar_fig = create_bar_chart(predictions)
                    st.plotly_chart(bar_fig, use_container_width=True)
                
                # Detailed results
                st.subheader("Detailed Predictions")
                
                for indicator, description in config.MARKET_INDICATORS.items():
                    score = predictions.get(indicator, 0)
                    
                    # Color coding
                    if score > 2:
                        color = "🟢"
                        sentiment = "Positive"
                    elif score < -2:
                        color = "🔴"
                        sentiment = "Negative"
                    else:
                        color = "🟡"
                        sentiment = "Neutral"
                    
                    st.metric(
                        label=f"{color} {description}",
                        value=f"{score:.2f}",
                        delta=f"{sentiment} impact"
                    )
                
                # Summary
                st.subheader("Market Summary")
                summary = st.session_state.predictor.get_market_summary(predictions)
                st.text(summary)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    
    elif submitted and not title:
        st.warning("Please enter at least an article title.")

def batch_prediction_page():
    """Batch prediction page"""
    st.header("📊 Batch Prediction")
    st.markdown("Upload a CSV file with multiple articles for batch prediction:")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="CSV should contain columns: title, content, description, source, language"
    )
    
    # Sample data download
    if st.button("Download Sample CSV Template"):
        sample_data = pd.DataFrame({
            'title': [
                'Turkish Central Bank raises interest rates',
                'Turkish exports reach record high',
                'Political tensions affect markets'
            ],
            'content': [
                'The central bank announced rate hikes...',
                'Export figures show strong performance...',
                'Political uncertainty creates volatility...'
            ],
            'description': [
                'Rate hike announcement',
                'Export performance data',
                'Political impact on markets'
            ],
            'source': ['Reuters', 'Bloomberg', 'CNN'],
            'language': ['en', 'en', 'en']
        })
        
        csv = sample_data.to_csv(index=False)
        st.download_button(
            label="Download CSV Template",
            data=csv,
            file_name="sample_articles.csv",
            mime="text/csv"
        )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_columns = ['title']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                return
            
            # Fill missing columns
            if 'content' not in df.columns:
                df['content'] = ''
            if 'description' not in df.columns:
                df['description'] = ''
            if 'source' not in df.columns:
                df['source'] = 'Unknown'
            if 'language' not in df.columns:
                df['language'] = 'en'
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(df.head())
            
            if st.button("Run Batch Prediction"):
                with st.spinner(f"Processing {len(df)} articles..."):
                    try:
                        # Make predictions
                        results_df = st.session_state.predictor.predict_batch(df)
                        
                        # Display results
                        st.success(f"Processed {len(results_df)} articles successfully!")
                        
                        # Summary statistics
                        st.subheader("Prediction Summary")
                        
                        pred_columns = [f'{indicator}_impact_pred' for indicator in config.MARKET_INDICATORS.keys()]
                        summary_stats = results_df[pred_columns].describe()
                        st.dataframe(summary_stats)
                        
                        # Visualization
                        st.subheader("Impact Distribution")
                        
                        for indicator in config.MARKET_INDICATORS.keys():
                            col_name = f'{indicator}_impact_pred'
                            if col_name in results_df.columns:
                                fig = px.histogram(
                                    results_df, 
                                    x=col_name,
                                    title=f"{config.MARKET_INDICATORS[indicator]} Impact Distribution",
                                    nbins=20
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results CSV",
                            data=csv_results,
                            file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")

def live_news_page():
    """Live news analysis page"""
    st.header("📡 Live News Analysis")
    st.markdown("Fetch and analyze recent news articles:")
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        days_back = st.slider("Days to look back", 1, 7, 3)
        max_articles = st.slider("Maximum articles to analyze", 5, 50, 20)
    
    with col2:
        language_filter = st.selectbox("Language filter", ["All", "English", "Turkish"])
        source_filter = st.selectbox("Source filter", ["All"] + list(config.NEWS_SOURCES['english']) + list(config.NEWS_SOURCES['turkish']))
    
    if st.button("Fetch and Analyze Recent News"):
        with st.spinner("Fetching recent news..."):
            try:
                # Fetch news
                fetcher = NewsFetcher()
                news_df = fetcher.fetch_all_news(days_back=days_back)
                
                if news_df.empty:
                    st.warning("No news articles found.")
                    return
                
                # Apply filters
                if language_filter != "All":
                    lang_code = "en" if language_filter == "English" else "tr"
                    news_df = news_df[news_df['language'] == lang_code]
                
                if source_filter != "All":
                    news_df = news_df[news_df['source'] == source_filter]
                
                # Limit number of articles
                news_df = news_df.head(max_articles)
                
                st.success(f"Found {len(news_df)} articles. Analyzing...")
                
                # Make predictions
                with st.spinner("Analyzing market impact..."):
                    results_df = st.session_state.predictor.predict_batch(news_df)
                
                # Display results
                st.subheader("Recent News Impact Analysis")
                
                # Show articles with highest impact
                pred_columns = [f'{indicator}_impact_pred' for indicator in config.MARKET_INDICATORS.keys()]
                results_df['avg_impact'] = results_df[pred_columns].abs().mean(axis=1)
                top_impact = results_df.nlargest(5, 'avg_impact')
                
                for idx, row in top_impact.iterrows():
                    with st.expander(f"📰 {row['title'][:100]}..."):
                        st.write(f"**Source:** {row['source']}")
                        st.write(f"**Published:** {row['published_at']}")
                        st.write(f"**Language:** {row['language']}")
                        
                        # Impact scores
                        impacts = {indicator: row[f'{indicator}_impact_pred'] for indicator in config.MARKET_INDICATORS.keys()}
                        
                        cols = st.columns(len(config.MARKET_INDICATORS))
                        for i, (indicator, description) in enumerate(config.MARKET_INDICATORS.items()):
                            with cols[i]:
                                score = impacts[indicator]
                                color = "normal" if abs(score) < 2 else ("inverse" if score < 0 else "off")
                                st.metric(description.split()[0], f"{score:.2f}")
                
                # Overall market sentiment
                st.subheader("Overall Market Sentiment")
                avg_impacts = {indicator: results_df[f'{indicator}_impact_pred'].mean() 
                              for indicator in config.MARKET_INDICATORS.keys()}
                
                sentiment_fig = create_bar_chart(avg_impacts)
                st.plotly_chart(sentiment_fig, use_container_width=True)
                
                # Download results
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Analysis Results",
                    data=csv_results,
                    file_name=f"live_news_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Live news analysis failed: {e}")

def model_info_page():
    """Model information page"""
    st.header("🤖 Model Information")
    
    try:
        # Model metadata
        if hasattr(st.session_state.predictor, 'metadata') and st.session_state.predictor.metadata:
            metadata = st.session_state.predictor.metadata
            
            st.subheader("Model Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Type", metadata.get('model_name', 'Unknown'))
                st.metric("Feature Count", metadata.get('feature_count', 'Unknown'))
            
            with col2:
                if 'metrics' in metadata:
                    test_metrics = metadata['metrics'].get('test_metrics', {})
                    st.metric("R² Score", f"{test_metrics.get('r2', 0):.4f}")
                    st.metric("MAE", f"{test_metrics.get('mae', 0):.4f}")
        
        # Target indicators
        st.subheader("Target Market Indicators")
        
        for indicator, description in config.MARKET_INDICATORS.items():
            st.write(f"**{indicator}**: {description}")
        
        # Feature information
        st.subheader("Feature Types")
        st.write("""
        The model uses the following types of features:
        - **TF-IDF Features**: Text content vectorization (up to 10,000 features)
        - **Text Length Features**: Article and title length
        - **Sentiment Features**: Positive/negative keyword counts
        - **Source Features**: News source reliability scores
        - **Temporal Features**: Publication time and day of week
        - **Language Features**: Language indicators
        """)
        
        # Model performance
        st.subheader("Model Performance Guidelines")
        st.write("""
        **Impact Score Interpretation:**
        - **+7 to +10**: Very strong positive impact
        - **+3 to +7**: Strong positive impact
        - **+1 to +3**: Moderate positive impact
        - **-1 to +1**: Minimal/neutral impact
        - **-3 to -1**: Moderate negative impact
        - **-7 to -3**: Strong negative impact
        - **-10 to -7**: Very strong negative impact
        
        **Note**: Predictions are based on historical patterns and should be used as guidance only.
        """)
        
    except Exception as e:
        st.error(f"Error loading model information: {e}")

if __name__ == "__main__":
    main()