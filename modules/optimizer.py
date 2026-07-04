"""
Resource Allocation Optimizer for HPD, DSNY, NYPD.
"""
import numpy as np
import pandas as pd

DEFAULT_TOTAL_RESOURCES = 500  # total officer/inspector equivalents


class ResourceOptimizer:
    """Allocate resources across agencies and boroughs."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() if not df.empty else pd.DataFrame()

    def get_workload_by_agency(self) -> pd.DataFrame:
        """Count open/pending complaints per agency."""
        if self.df.empty or 'agency' not in self.df.columns:
            return pd.DataFrame()
        open_df = self.df[self.df['status'].isin(['Open', 'In Progress',
                                                   'Assigned'])]
        return open_df.groupby('agency').size().reset_index(name='open_count')

    def get_workload_by_borough(self) -> pd.DataFrame:
        if self.df.empty or 'borough' not in self.df.columns:
            return pd.DataFrame()
        open_df = self.df[self.df['status'].isin(['Open', 'In Progress',
                                                   'Assigned'])]
        return open_df.groupby('borough').size().reset_index(name='open_count')

    def optimize_allocation(self,
                            total_resources: int = DEFAULT_TOTAL_RESOURCES
                            ) -> dict:
        """Proportionally allocate resources to HPD, DSNY, NYPD
        based on current open complaint volume.

        Returns dict with agency → {resources, open_complaints, ratio}.
        """
        workload = self.get_workload_by_agency()
        if workload.empty:
            return {}

        # Focus on top 3 agencies
        top3 = ['HPD', 'DSNY', 'NYPD']
        wl = workload[workload['agency'].isin(top3)].copy()
        total_open = wl['open_count'].sum()

        result = {}
        for _, row in wl.iterrows():
            agency = row['agency']
            count = row['open_count']
            share = count / total_open if total_open else 1 / len(top3)
            res = max(int(share * total_resources), 5)
            result[agency] = {
                'resources': res,
                'open_complaints': count,
                'ratio': round(count / res, 2) if res else 0,
                'share_pct': round(share * 100, 1)
            }

        # Remaining resources to other agencies
        used = sum(v['resources'] for v in result.values())
        remaining = total_resources - used
        other = workload[~workload['agency'].isin(top3)]
        if not other.empty and remaining > 0:
            other_total = other['open_count'].sum()
            for _, row in other.iterrows():
                share = row['open_count'] / other_total if other_total else 0
                res = max(int(share * remaining), 2)
                result[row['agency']] = {
                    'resources': res,
                    'open_complaints': row['open_count'],
                    'ratio': round(row['open_count'] / res, 2) if res else 0,
                    'share_pct': round(share * remaining /
                                       total_resources * 100, 1)
                }
        return result

    def suggest_reallocation(self) -> list:
        """Identify overloaded / underloaded agencies and suggest shifts."""
        alloc = self.optimize_allocation()
        if not alloc:
            return []
        avg_ratio = np.mean([v['ratio'] for v in alloc.values()])
        suggestions = []
        for agency, info in alloc.items():
            if info['ratio'] > avg_ratio * 1.5:
                suggestions.append(
                    f"⚠️ {agency} is overloaded "
                    f"(ratio {info['ratio']:.1f} vs avg {avg_ratio:.1f}). "
                    f"Consider moving {max(info['resources']//5, 2)} "
                    f"resources from a lighter agency."
                )
            elif info['ratio'] < avg_ratio * 0.5:
                suggestions.append(
                    f"💡 {agency} has spare capacity "
                    f"(ratio {info['ratio']:.1f} vs avg {avg_ratio:.1f}). "
                    f"Can lend resources to busier agencies."
                )
        return suggestions