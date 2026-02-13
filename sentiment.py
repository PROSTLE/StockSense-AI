
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

_vader = SentimentIntensityAnalyzer()

MODEL_NAME = "ProsusAI/finbert"
_tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
_finbert = BertForSequenceClassification.from_pretrained(MODEL_NAME)
_finbert.eval()
FINBERT_LABELS = ["negative", "neutral", "positive"]

VADER_WEIGHT = 0.4
FINBERT_WEIGHT = 0.6


def _score_vader(text: str) -> dict:
    scores = _vader.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    confidence = round(abs(compound), 4)
    return {"label": label, "score": round(compound, 4), "confidence": confidence}


def _score_finbert(text: str) -> dict:
    tokens = _tokenizer(text, return_tensors="pt", padding=True,
                        truncation=True, max_length=128)
    with torch.no_grad():
        logits = _finbert(**tokens).logits
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().tolist()
    idx = int(np.argmax(probs))

    signed_score = probs[2] - probs[0]
    return {
        "label": FINBERT_LABELS[idx],
        "score": round(signed_score, 4),
        "confidence": round(probs[idx], 4),
    }


def _score_text(text: str) -> dict:
    vader = _score_vader(text)
    finbert = _score_finbert(text)

    combined_score = (VADER_WEIGHT * vader["score"]) + (FINBERT_WEIGHT * finbert["score"])

    if combined_score > 0.1:
        label = "positive"
    elif combined_score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    combined_confidence = (VADER_WEIGHT * vader["confidence"]) + (FINBERT_WEIGHT * finbert["confidence"])

    return {
        "text": text,
        "label": label,
        "confidence": round(combined_confidence, 4),
        "combined_score": round(combined_score, 4),
        "vader": vader,
        "finbert": finbert,
    }


def fetch_news_headlines(ticker: str, max_items: int = 10) -> list[str]:
    stock = yf.Ticker(ticker)
    news = stock.news or []
    headlines = []
    for item in news[:max_items]:
        content = item.get("content", {})
        title = content.get("title", "") or item.get("title", "")
        if title:
            headlines.append(title)
    return headlines


def analyze_sentiment(ticker: str) -> dict:
    headlines = fetch_news_headlines(ticker)
    if not headlines:
        return {
            "ticker": ticker,
            "articles_analyzed": 0,
            "details": [],
            "overall_sentiment": "neutral",
            "overall_score": 0.0,
            "method": "VADER (40%) + FinBERT (60%) ensemble",
        }

    details = [_score_text(h) for h in headlines]

    avg = float(np.mean([d["combined_score"] for d in details]))

    if avg > 0.15:
        overall = "positive"
    elif avg < -0.15:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "ticker": ticker,
        "articles_analyzed": len(details),
        "details": details,
        "overall_sentiment": overall,
        "overall_score": round(avg, 4),
        "method": "VADER (40%) + FinBERT (60%) ensemble",
    }
