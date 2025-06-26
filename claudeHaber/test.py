from predict import MarketImpactPredictor

predictor = MarketImpactPredictor()
predictor.load_model()

predictions = predictor.predict_single_article(
    title="Turkish Central Bank raises interest rates",
    content="The central bank announced a rate hike to combat inflation...",
    source="Reuters",
    language="en"
)

print(predictions)
# Output: {'USD_TRY': -2.3, 'GAU_TRY': 1.5, 'BIST100': 3.2, 'BTC_TRY': 0.8}