"""
Decision Recommendation Engine: priority scoring and action plans.
"""
import numpy as np
import pandas as pd

from .nlp_engine import NLPEngine

# Pre-defined severity weights per complaint type (1-10)
SEVERITY_WEIGHTS = {
    'HEAT/HOT WATER': 9, 'ELECTRIC': 9, 'WATER LEAK': 8,
    'Sewer': 8, 'Food Poisoning': 8, 'Elevator': 7,
    'DOOR/WINDOW': 6, 'PLUMBING': 6, 'APPLIANCE': 5,
    'FLOORING/STAIRS': 5, 'PAINT/PLASTER': 4,
    'Noise - Residential': 5, 'Noise - Commercial': 5,
    'Noise - Vehicle': 4, 'Illegal Parking': 4,
    'Blocked Driveway': 5, 'Snow or Ice': 6,
    'Missed Collection': 3, 'Dirty Condition': 3,
    'Homeless Person Assistance': 6, 'Street Light Condition': 4,
    'Taxi Complaint': 3
}

# Borough vulnerability index (higher = more underserved)
BOROUGH_VULNERABILITY = {
    'BRONX': 1.3, 'BROOKLYN': 1.1, 'MANHATTAN': 0.9,
    'QUEENS': 1.0, 'STATEN ISLAND': 1.2
}


class DecisionEngine:
    """Generate priority scores and decision recommendations."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() if not df.empty else pd.DataFrame()
        self.nlp = NLPEngine()

    def calculate_priority_score(self, row: pd.Series) -> dict:
        """Calculate a composite priority score (0-100) for one complaint.

        Components:
          - Severity weight (0-30)
          - Urgency score  (0-30)
          - Sentiment      (0-20)
          - Borough vuln.  (0-10)
          - Flag penalty   (0-10)
        """
        ctype = row.get('complaint_type', '')
        borough = row.get('borough', '')
        desc = str(row.get('descriptor', '')) + ' ' + \
               str(row.get('descriptor_2', ''))
        desc = desc.strip()

        # Severity
        severity = SEVERITY_WEIGHTS.get(ctype, 5)
        severity_pts = (severity / 10) * 30

        # Urgency from NLP
        urg = self.nlp.detect_urgency(desc, ctype)
        urgency_pts = (urg['score'] / 10) * 30

        # Sentiment
        sent = self.nlp.analyze_sentiment(desc)
        sentiment_pts = max(0, -sent['polarity']) * 20  # more negative → higher

        # Borough vulnerability
        vuln = BOROUGH_VULNERABILITY.get(borough, 1.0)
        borough_pts = (vuln - 0.8) / 0.6 * 10  # normalise to ~0-10

        # Flag penalty
        flag_pts = 0
        if str(row.get('is_flagged', '')).lower() == 'true':
            flag_pts = min(row.get('flag_reason', '') and 5 or 3, 10)

        total = min(round(severity_pts + urgency_pts +
                          sentiment_pts + borough_pts + flag_pts, 1), 100)

        if total >= 75:
            priority = '🔴 Critical'
        elif total >= 55:
            priority = '🟠 High'
        elif total >= 35:
            priority = '🟡 Medium'
        else:
            priority = '🟢 Low'

        return {
            'score': total,
            'priority': priority,
            'breakdown': {
                'severity': round(severity_pts, 1),
                'urgency': round(urgency_pts, 1),
                'sentiment': round(sentiment_pts, 1),
                'borough_vulnerability': round(borough_pts, 1),
                'flag_penalty': round(flag_pts, 1)
            }
        }

    def score_all(self) -> pd.DataFrame:
        """Add priority columns to every row in the dataset."""
        if self.df.empty:
            return self.df
        results = self.df.apply(self.calculate_priority_score, axis=1)
        self.df['priority_score'] = results.apply(lambda r: r['score'])
        self.df['priority_label'] = results.apply(lambda r: r['priority'])
        return self.df

    def generate_recommendations(self, top_n: int = 10) -> list:
        """Top-N highest-priority complaints with recommended actions."""
        scored = self.score_all()
        if scored.empty:
            return []
        top = scored.nlargest(top_n, 'priority_score')
        recs = []
        for _, row in top.iterrows():
            action = self._recommend_action(row)
            recs.append({
                'unique_key': row.get('unique_key', ''),
                'complaint_type': row.get('complaint_type', ''),
                'borough': row.get('borough', ''),
                'agency': row.get('agency', ''),
                'priority_score': row.get('priority_score', 0),
                'priority_label': row.get('priority_label', ''),
                'action': action
            })
        return recs

    @staticmethod
    def _recommend_action(row: pd.Series) -> str:
        """Generate a text recommendation for one complaint."""
        ctype = row.get('complaint_type', '')
        agency = row.get('agency', '')
        borough = row.get('borough', '')
        score = row.get('priority_score', 0)

        if score >= 75:
            prefix = "🚨 IMMEDIATE dispatch required."
        elif score >= 55:
            prefix = "⚡ Expedite within 4 hours."
        elif score >= 35:
            prefix = "📋 Schedule within 24 hours."
        else:
            prefix = "📝 Routine scheduling."

        return (f"{prefix} Route to {agency} unit in {borough}. "
                f"Complaint type: {ctype}.")