"""
Analytics: trend data, distributions, crisis early warning, predictions.
"""
import numpy as np
import pandas as pd


class AnalyticsEngine:
    """Statistical analytics and prediction engine."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() if not df.empty else pd.DataFrame()

    # ── Helpers ────────────────────────────────────────────────────────
    def _ensure_date(self) -> pd.DataFrame:
        df = self.df.copy()
        if 'created_date' in df.columns:
            df['created_date'] = pd.to_datetime(df['created_date'],
                                                errors='coerce')
        return df

    # ── Trend data ─────────────────────────────────────────────────────
    def get_trend_data(self) -> pd.DataFrame:
        """Daily complaint counts for trend chart."""
        df = self._ensure_date()
        if df.empty or 'created_date' not in df.columns:
            return pd.DataFrame(columns=['date', 'count'])
        daily = df.groupby(df['created_date'].dt.date).size()\
                  .reset_index(name='count')
        daily.columns = ['date', 'count']
        return daily

    # ── Distributions ──────────────────────────────────────────────────
    def get_borough_distribution(self) -> pd.DataFrame:
        if self.df.empty or 'borough' not in self.df.columns:
            return pd.DataFrame(columns=['borough', 'count'])
        return self.df['borough'].value_counts().reset_index()
    
    def get_agency_distribution(self) -> pd.DataFrame:
        if self.df.empty or 'agency' not in self.df.columns:
            return pd.DataFrame(columns=['agency', 'count'])
        return self.df['agency'].value_counts().reset_index()

    def get_complaint_type_distribution(self, top_n: int = 15) -> pd.DataFrame:
        if self.df.empty or 'complaint_type' not in self.df.columns:
            return pd.DataFrame(columns=['complaint_type', 'count'])
        return self.df['complaint_type'].value_counts().head(top_n).reset_index()

    def get_status_distribution(self) -> pd.DataFrame:
        if self.df.empty or 'status' not in self.df.columns:
            return pd.DataFrame(columns=['status', 'count'])
        return self.df['status'].value_counts().reset_index()

    # ── Crisis early warning ───────────────────────────────────────────
    def detect_crisis(self, threshold_std: float = 2.0) -> list:
        """Detect days where complaint volume exceeds mean + k*std.
        Returns a list of alert dicts."""
        trend = self.get_trend_data()
        if trend.empty or len(trend) < 3:
            return []
        mean_cnt = trend['count'].mean()
        std_cnt = trend['count'].std()
        spike = threshold_std * std_cnt

        alerts = []
        for _, row in trend.iterrows():
            if row['count'] > mean_cnt + spike:
                alerts.append({
                    'date': str(row['date']),
                    'count': int(row['count']),
                    'expected': round(mean_cnt, 1),
                    'deviation': round((row['count'] - mean_cnt) / std_cnt, 2)
                })
        return alerts

    # ── Simple prediction ──────────────────────────────────────────────
    def predict_next_week(self) -> dict:
        """Simple moving-average prediction for the next 7 days."""
        trend = self.get_trend_data()
        if trend.empty:
            return {'predicted_daily': 0, 'predicted_weekly': 0,
                    'confidence': 'low'}
        recent = trend.tail(7)
        avg = recent['count'].mean()
        std = recent['count'].std()
        confidence = 'high' if std < avg * 0.3 else \
                     'medium' if std < avg * 0.6 else 'low'
        return {
            'predicted_daily': round(avg, 1),
            'predicted_weekly': round(avg * 7, 1),
            'confidence': confidence
        }

    # ── Resolution time stats ──────────────────────────────────────────
    def get_resolution_stats(self) -> dict:
        """Average resolution time (in hours) by agency."""
        df = self._ensure_date()
        if df.empty or 'closed_date' not in df.columns:
            return {}
        df['closed_date'] = pd.to_datetime(df['closed_date'], errors='coerce')
        valid = df.dropna(subset=['created_date', 'closed_date', 'agency'])
        if valid.empty:
            return {}
        valid['resolution_hours'] = (
            valid['closed_date'] - valid['created_date']
        ).dt.total_seconds() / 3600
        valid = valid[valid['resolution_hours'] >= 0]
        return valid.groupby('agency')['resolution_hours'].agg(
            ['mean', 'median', 'count']
        ).round(1).to_dict('index')