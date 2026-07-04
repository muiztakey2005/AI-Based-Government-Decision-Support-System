"""
NLP engine: sentiment analysis, urgency detection, keyword extraction.
Uses TextBlob with a rule-based fallback.
"""
import re

# Try to import TextBlob; fall back gracefully
try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False

# ── Keyword lists ─────────────────────────────────────────────────────────
HIGH_URGENCY_WORDS = [
    'emergency', 'urgent', 'danger', 'fire', 'flood', 'collapse',
    'explosion', 'gas leak', 'electrocution', 'shooting', 'stabbing',
    'assault', 'break-in', 'robbery', 'dead', 'dying', 'trapped',
    'structural', 'hazardous', 'toxic', 'chemical'
]
MEDIUM_URGENCY_WORDS = [
    'broken', 'damaged', 'leaking', 'no heat', 'no hot water', 'no water',
    'no electricity', 'outage', 'blocked', 'overflow', 'infestation',
    'mold', 'asbestos', 'stuck', 'crack', 'hole'
]
NEGATIVE_WORDS = [
    'terrible', 'horrible', 'awful', 'worst', 'disgusting', 'unacceptable',
    'frustrated', 'angry', 'furious', 'disappointed', 'complaint',
    'problem', 'issue', 'fail', 'never', 'refuse'
]


class NLPEngine:
    """NLP analysis for complaint text."""

    @staticmethod
    def analyze_sentiment(text: str) -> dict:
        """Return sentiment label, polarity, and subjectivity."""
        if HAS_TEXTBLOB:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
        else:
            # Simple rule-based fallback
            text_lower = text.lower()
            neg_count = sum(1 for w in NEGATIVE_WORDS if w in text_lower)
            polarity = -min(neg_count * 0.25, 1.0)
            subjectivity = min(neg_count * 0.15 + 0.3, 1.0)

        if polarity <= -0.5:
            label = 'Very Negative 😠'
        elif polarity <= -0.1:
            label = 'Negative 😞'
        elif polarity < 0.1:
            label = 'Neutral 😐'
        elif polarity < 0.5:
            label = 'Positive 🙂'
        else:
            label = 'Very Positive 😄'

        return {
            'label': label,
            'polarity': round(polarity, 3),
            'subjectivity': round(subjectivity, 3)
        }

    @staticmethod
    def detect_urgency(text: str, complaint_type: str = '') -> dict:
        """Return urgency level and score (1-10)."""
        text_lower = text.lower()
        score = 3  # baseline

        high_hits = sum(1 for w in HIGH_URGENCY_WORDS if w in text_lower)
        med_hits = sum(1 for w in MEDIUM_URGENCY_WORDS if w in text_lower)
        score += high_hits * 3 + med_hits * 1

        # Certain complaint types are inherently urgent
        high_types = ['HEAT/HOT WATER', 'ELECTRIC', 'Sewer', 'Food Poisoning']
        med_types = ['WATER LEAK', 'Noise - Commercial', 'Illegal Parking',
                     'Blocked Driveway']
        if complaint_type in high_types:
            score += 3
        elif complaint_type in med_types:
            score += 1

        score = min(max(score, 1), 10)

        if score >= 8:
            level = '🔴 Critical'
        elif score >= 6:
            level = '🟠 High'
        elif score >= 4:
            level = '🟡 Medium'
        else:
            level = '🟢 Low'

        return {'level': level, 'score': score}

    @staticmethod
    def extract_keywords(text: str, top_n: int = 8) -> list:
        """Simple keyword extraction: remove stop-words, return top nouns/
        meaningful words by frequency."""
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'need', 'dare', 'ought', 'used', 'to', 'of', 'in', 'for',
            'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between',
            'out', 'off', 'over', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'because', 'but',
            'and', 'or', 'if', 'while', 'about', 'up', 'it', 'its',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'they', 'them', 'their', 'this', 'that',
            'these', 'those', 'what', 'which', 'who', 'whom', 'am'
        }
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        words = [w for w in words if w not in stop_words]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return sorted(freq, key=freq.get, reverse=True)[:top_n]

    def analyze_complaint(self, text: str,
                          complaint_type: str = '') -> dict:
        """Full NLP analysis of a complaint."""
        sentiment = self.analyze_sentiment(text)
        urgency = self.detect_urgency(text, complaint_type)
        keywords = self.extract_keywords(text)
        return {
            'sentiment': sentiment,
            'urgency': urgency,
            'keywords': keywords
        }