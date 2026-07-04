"""
Policy Impact Simulator: model how budget changes affect complaint resolution.
"""
import numpy as np
import pandas as pd

# Baseline performance metrics per agency (mock but realistic)
BASELINE = {
    'HPD':   {'resolution_rate': 0.72, 'avg_hours': 48,  'budget_share': 0.35},
    'DSNY':  {'resolution_rate': 0.85, 'avg_hours': 24,  'budget_share': 0.25},
    'NYPD':  {'resolution_rate': 0.68, 'avg_hours': 4,   'budget_share': 0.30},
    'OTHER': {'resolution_rate': 0.60, 'avg_hours': 72,  'budget_share': 0.10},
}

ELASTICITY = {
    # % improvement in resolution_rate per 10 % budget increase
    'HPD':   0.04,
    'DSNY':  0.03,
    'NYPD':  0.05,
    'OTHER': 0.03,
}


class PolicySimulator:
    """Simulate the impact of budget allocation changes on KPIs."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() if not df.empty else pd.DataFrame()

    def simulate_budget(self, budget_pct: dict) -> dict:
        """Given a budget allocation dict {agency: pct_of_total_budget},
        return projected KPIs.

        budget_pct example: {'HPD': 40, 'DSNY': 25, 'NYPD': 30, 'OTHER': 5}
        Sums to 100.
        """
        results = {}
        for agency, new_pct in budget_pct.items():
            base = BASELINE.get(agency, BASELINE['OTHER'])
            elast = ELASTICITY.get(agency, ELASTICITY['OTHER'])
            change = (new_pct - base['budget_share'] * 100) / 10  # in 10% units
            # Projected resolution rate
            proj_rate = base['resolution_rate'] + elast * change
            proj_rate = max(0.1, min(proj_rate, 0.99))
            # Projected avg resolution time (inverse relationship)
            time_factor = 1 - (elast * change * 0.5)
            proj_hours = base['avg_hours'] * max(time_factor, 0.3)
            results[agency] = {
                'current_resolution_rate': round(base['resolution_rate'] * 100, 1),
                'projected_resolution_rate': round(proj_rate * 100, 1),
                'current_avg_hours': base['avg_hours'],
                'projected_avg_hours': round(proj_hours, 1),
                'budget_pct': new_pct,
                'change_pct': round(change * 10, 1)
            }
        return results

    def simulate_policy_change(self, policy: str, intensity: float = 1.0
                               ) -> dict:
        """Simulate a named policy change.

        Policies:
          - 'proactive_patrols'    → NYPD resolution rate up, avg hours down
          - 'housing_inspection'   → HPD resolution rate up
          - 'sanitation_blitz'     → DSNY resolution rate up, hours down
        """
        impact = {}
        if policy == 'proactive_patrols':
            impact['NYPD'] = {
                'resolution_rate_change': round(5 * intensity, 1),
                'avg_hours_change': round(-1.5 * intensity, 1)
            }
        elif policy == 'housing_inspection':
            impact['HPD'] = {
                'resolution_rate_change': round(8 * intensity, 1),
                'avg_hours_change': round(-12 * intensity, 1)
            }
        elif policy == 'sanitation_blitz':
            impact['DSNY'] = {
                'resolution_rate_change': round(6 * intensity, 1),
                'avg_hours_change': round(-8 * intensity, 1)
            }
        return impact