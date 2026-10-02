"""
News sentiment scoring using VADER (lightweight, no API key needed).

For production use, replace VADER with FinBERT (ProsusAI/finbert) —
a BERT model fine-tuned on financial text for much better accuracy.

    from transformers import pipeline
    pipe = pipeline("text-classification", model="ProsusAI/finbert")
    result = pipe("Apple reports record quarterly revenue")
    # → [{'label': 'positive', 'score': 0.97}]

VADER is used here to keep the project dependency-light and runnable
without GPU. The integration point is identical — swap score_headline().
"""

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _analyzer = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except ImportError:
    _analyzer = None
    VADER_AVAILABLE = False
    print("⚠️  vaderSentiment not installed. Run: pip install vaderSentiment")


def score_headline(text: str) -> float:
    """Score a single headline. Returns compound: -1.0 (bearish) to +1.0 (bullish)."""
    if not VADER_AVAILABLE or _analyzer is None:
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]


def score_headlines(headlines: list) -> float:
    """Average sentiment over a list of headlines."""
    if not headlines:
        return 0.0
    return sum(score_headline(h) for h in headlines) / len(headlines)


def sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "Bullish"
    elif score <= -0.05:
        return "Bearish"
    return "Neutral"


# ── Demo ─────────────────────────────────────────────────────────────────────
DEMO_HEADLINES = {
    "bullish": [
        "Apple reports record-breaking quarterly revenue",
        "JPMorgan profits surge as interest income rises",
        "Microsoft Azure cloud growth accelerates beyond expectations",
    ],
    "bearish": [
        "Fed signals prolonged rate hikes amid persistent inflation",
        "Goldman Sachs announces major layoffs amid revenue decline",
        "Tech sector faces mounting regulatory pressure from EU",
    ],
    "neutral": [
        "Markets open mixed ahead of Fed announcement",
        "Investors await quarterly earnings reports",
    ],
}

if __name__ == "__main__":
    print("📰 Sentiment scoring demo (VADER):\n")
    for category, headlines in DEMO_HEADLINES.items():
        score = score_headlines(headlines)
        print(f"  {category.upper():>8}: {score:+.3f} → {sentiment_label(score)}")
        for h in headlines:
            print(f"           • {h}")
        print()
