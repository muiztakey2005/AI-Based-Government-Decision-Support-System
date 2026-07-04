"""
AI Copilot: rule-based natural language query engine over complaint data.
"""
import re

import pandas as pd


class AICopilot:
    """Answer natural-language questions about the complaint dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() if not df.empty else pd.DataFrame()

    # ── Entity extraction ──────────────────────────────────────────────
    @staticmethod
    def _find_borough(text: str) -> str:
        for b in ['MANHATTAN', 'BROOKLYN', 'BRONX', 'QUEENS',
                   'STATEN ISLAND']:
            if b.lower() in text.lower():
                return b
        return ''

    @staticmethod
    def _find_agency(text: str) -> str:
        for a in ['NYPD', 'HPD', 'DSNY', 'DOT', 'DHS', 'DOB',
                   'DOHMH', 'TLC', 'DEP']:
            if a.lower() in text.lower():
                return a
        return ''

    def _find_complaint_type(self, text: str) -> str:
        if self.df.empty or 'complaint_type' not in self.df.columns:
            return ''
        for ct in self.df['complaint_type'].unique():
            if ct.lower() in text.lower():
                return ct
        return ''

    # ── Query processing ───────────────────────────────────────────────
    def process_query(self, query: str) -> str:
        """Main entry point – returns a formatted text answer."""
        q = query.lower().strip()

        # ── Total complaints ───────────────────────────────────────────
        if any(w in q for w in ['how many', 'total', 'count', 'number']):
            return self._answer_count(q)

        # ── Borough-specific ───────────────────────────────────────────
        if any(w in q for w in ['borough', 'which borough', 'where']):
            return self._answer_borough(q)

        # ── Agency-specific ────────────────────────────────────────────
        if any(w in q for w in ['agency', 'department', 'who handles']):
            return self._answer_agency(q)

        # ── Trend / comparison ─────────────────────────────────────────
        if any(w in q for w in ['trend', 'over time', 'increase',
                                'decrease', 'compare']):
            return self._answer_trend(q)

        # ── Status ─────────────────────────────────────────────────────
        if 'status' in q or 'open' in q or 'closed' in q:
            return self._answer_status(q)

        # ── Top / most common ──────────────────────────────────────────
        if any(w in q for w in ['top', 'most common', 'most frequent',
                                'popular', 'biggest']):
            return self._answer_top(q)

        # ── Recommendation ─────────────────────────────────────────────
        if any(w in q for w in ['recommend', 'suggest', 'what should',
                                'action', 'priority']):
            return self._answer_recommendation(q)

        # ── Fallback ───────────────────────────────────────────────────
        return self._answer_summary()

    # ── Answer helpers ─────────────────────────────────────────────────
    def _answer_count(self, q: str) -> str:
        total = len(self.df)
        borough = self._find_borough(q)
        agency = self._find_agency(q)
        ctype = self._find_complaint_type(q)

        filtered = self.df.copy()
        if borough:
            filtered = filtered[filtered['borough'] == borough]
        if agency:
            filtered = filtered[filtered['agency'] == agency]
        if ctype:
            filtered = filtered[filtered['complaint_type'] == ctype]

        parts = [f"There are **{len(filtered):,}** complaints"]
        if borough:
            parts.append(f"in {borough}")
        if agency:
            parts.append(f"handled by {agency}")
        if ctype:
            parts.append(f"of type '{ctype}'")
        return ' '.join(parts) + f" (out of {total:,} total)."

    def _answer_borough(self, q: str) -> str:
        if self.df.empty or 'borough' not in self.df.columns:
            return "No borough data available."
        counts = self.df['borough'].value_counts()
        lines = ["📊 **Complaints by Borough:**"]
        for b, c in counts.items():
            lines.append(f"  • {b}: {c:,}")
        return '\n'.join(lines)

    def _answer_agency(self, q: str) -> str:
        if self.df.empty or 'agency' not in self.df.columns:
            return "No agency data available."
        agency = self._find_agency(q)
        if agency:
            subset = self.df[self.df['agency'] == agency]
            return (f"🏢 **{agency}** has **{len(subset):,}** complaints. "
                    f"Top types: "
                    f"{', '.join(subset['complaint_type'].value_counts().head(3).index.tolist())}")
        counts = self.df['agency'].value_counts()
        lines = ["📊 **Complaints by Agency:**"]
        for a, c in counts.items():
            lines.append(f"  • {a}: {c:,}")
        return '\n'.join(lines)

    def _answer_trend(self, q: str) -> str:
        if self.df.empty or 'created_date' not in self.df.columns:
            return "No date data available for trend analysis."
        df = self.df.copy()
        df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
        df = df.dropna(subset=['created_date'])
        daily = df.groupby(df['created_date'].dt.date).size()
        if len(daily) < 2:
            return "Not enough data for trend analysis."
        recent = daily.tail(7).mean()
        earlier = daily.head(7).mean()
        change = ((recent - earlier) / earlier * 100) if earlier else 0
        direction = "📈 increasing" if change > 0 else "📉 decreasing"
        return (f"📈 **Trend Analysis:** Average daily complaints changed "
                f"from {earlier:.1f} to {recent:.1f} "
                f"({direction} by {abs(change):.1f}%).")

    def _answer_status(self, q: str) -> str:
        if self.df.empty or 'status' not in self.df.columns:
            return "No status data available."
        counts = self.df['status'].value_counts()
        lines = ["📋 **Complaint Status Breakdown:**"]
        for s, c in counts.items():
            lines.append(f"  • {s}: {c:,}")
        return '\n'.join(lines)

    def _answer_top(self, q: str) -> str:
        if self.df.empty or 'complaint_type' not in self.df.columns:
            return "No complaint type data available."
        top = self.df['complaint_type'].value_counts().head(5)
        lines = ["🏆 **Top 5 Complaint Types:**"]
        for i, (ct, c) in enumerate(top.items(), 1):
            lines.append(f"  {i}. {ct} — {c:,}")
        return '\n'.join(lines)

    def _answer_recommendation(self, q: str) -> str:
        if self.df.empty:
            return "No data available for recommendations."
        # Quick stat-based recommendation
        top_borough = self.df['borough'].value_counts().idxmax() \
            if 'borough' in self.df.columns else 'N/A'
        top_type = self.df['complaint_type'].value_counts().idxmax() \
            if 'complaint_type' in self.df.columns else 'N/A'
        open_count = len(self.df[self.df['status'].isin(
            ['Open', 'In Progress'])]) if 'status' in self.df.columns else 0
        return (f"💡 **Quick Recommendations:**\n"
                f"  • Focus resources on **{top_borough}** (highest volume)\n"
                f"  • Prioritize **{top_type}** complaints (most common)\n"
                f"  • {open_count:,} complaints still open — accelerate resolution")

    def _answer_summary(self) -> str:
        """General summary when intent is unclear."""
        total = len(self.df)
        if total == 0:
            return "No data loaded. Please check your dataset files."
        boroughs = self.df['borough'].nunique() \
            if 'borough' in self.df.columns else 0
        agencies = self.df['agency'].nunique() \
            if 'agency' in self.df.columns else 0
        open_cnt = len(self.df[self.df['status'].isin(
            ['Open', 'In Progress'])]) if 'status' in self.df.columns else 0
        return (f"📊 **Dataset Overview:** {total:,} complaints across "
                f"{boroughs} boroughs and {agencies} agencies. "
                f"{open_cnt:,} currently open. "
                f"Ask me about counts, trends, boroughs, agencies, or "
                f"recommendations!")