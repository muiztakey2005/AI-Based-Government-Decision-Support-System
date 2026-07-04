"""
Data loading, combining, cleaning, and fake-complaint detection.
Works with both the historical NYC 311 dataset and live citizen complaints.
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd

# ── Mapping: complaint_type → agency ──────────────────────────────────────
COMPLAINT_AGENCY_MAP = {
    'Noise - Commercial': 'NYPD', 'Noise - Residential': 'NYPD',
    'Noise - Vehicle': 'NYPD', 'Illegal Parking': 'NYPD',
    'Blocked Driveway': 'NYPD',
    'HEAT/HOT WATER': 'HPD', 'WATER LEAK': 'HPD', 'DOOR/WINDOW': 'HPD',
    'FLOORING/STAIRS': 'HPD', 'APPLIANCE': 'HPD', 'ELECTRIC': 'HPD',
    'PLUMBING': 'HPD', 'PAINT/PLASTER': 'HPD',
    'Snow or Ice': 'DSNY', 'Missed Collection': 'DSNY',
    'Dirty Condition': 'DSNY',
    'Homeless Person Assistance': 'DHS',
    'Street Light Condition': 'DOT',
    'Taxi Complaint': 'TLC',
    'Elevator': 'DOB',
    'Food Poisoning': 'DOHMH',
    'Sewer': 'DEP',
}

AGENCY_NAMES = {
    'NYPD': 'New York City Police Department',
    'HPD': 'Department of Housing Preservation and Development',
    'DSNY': 'Department of Sanitation',
    'DHS': 'Department of Homeless Services',
    'DOT': 'Department of Transportation',
    'TLC': 'Taxi and Limousine Commission',
    'DOB': 'Department of Buildings',
    'DOHMH': 'Department of Health and Mental Hygiene',
    'DEP': 'Department of Environmental Protection',
}

BOROUGHS = ['MANHATTAN', 'BROOKLYN', 'BRONX', 'QUEENS', 'STATEN ISLAND']

COMPLAINT_TYPES = sorted(COMPLAINT_AGENCY_MAP.keys())

LOCATION_TYPES = [
    'Residential Building/House', 'Store/Commercial', 'Street/Sidewalk',
    'Street', 'Sidewalk', 'Restaurant/Bar/Deli/Bakery', 'Park/Playground',
    'Club/Bar', 'Office Building', 'Other'
]


class DataProcessor:
    """Central data handler for the platform."""

    def __init__(self, hist_path='data/nyc_dataset.csv',
                 live_path='data/live_complaints.csv'):
        self.hist_path = hist_path
        self.live_path = live_path
        self._hist_df = None
        self._live_df = None
        self._combined_df = None
        self._borough_complaint_freq = None

    # ── Loading ────────────────────────────────────────────────────────
    def load_historical(self) -> pd.DataFrame:
        """Load the original NYC 311 dataset."""
        if self._hist_df is not None:
            return self._hist_df
        if not os.path.exists(self.hist_path):
            print(f"⚠️  {self.hist_path} not found – returning empty DataFrame")
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.hist_path, low_memory=False)
            # Drop any pure-index columns that pandas might pick up
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            self._hist_df = df
            return df
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return pd.DataFrame()

    def load_live(self) -> pd.DataFrame:
        """Load citizen-submitted complaints."""
        if self._live_df is not None:
            return self._live_df
        if not os.path.exists(self.live_path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.live_path)
            self._live_df = df
            return df
        except Exception as e:
            print(f"❌ Error loading live data: {e}")
            return pd.DataFrame()

    def get_combined(self) -> pd.DataFrame:
        """Return a single DataFrame that merges historical + live data."""
        if self._combined_df is not None:
            return self._combined_df
        hist = self.load_historical().copy()
        live = self.load_live().copy()
        # Ensure both have a 'source' tag
        hist['source'] = 'historical'
        if 'source' not in live.columns:
            live['source'] = 'live'
        # Extra columns in live but not in historical
        for col in ['citizen_name', 'citizen_mobile', 'is_flagged', 'flag_reason']:
            if col not in hist.columns:
                hist[col] = np.nan
        for col in hist.columns:
            if col not in live.columns:
                live[col] = np.nan
        common = list(set(hist.columns) & set(live.columns))
        self._combined_df = pd.concat([hist[common], live[common]],
                                      ignore_index=True)
        return self._combined_df

    def reload(self):
        """Force reload on next access (e.g. after a new complaint is saved)."""
        self._hist_df = None
        self._live_df = None
        self._combined_df = None
        self._borough_complaint_freq = None

    # ── Lookup helpers ─────────────────────────────────────────────────
    @staticmethod
    def get_boroughs() -> list:
        return BOROUGHS

    @staticmethod
    def get_complaint_types() -> list:
        return COMPLAINT_TYPES

    @staticmethod
    def get_location_types() -> list:
        return LOCATION_TYPES

    @staticmethod
    def get_agency_for_type(complaint_type: str) -> str:
        return COMPLAINT_AGENCY_MAP.get(complaint_type, 'NYPD')

    @staticmethod
    def get_agency_name(agency: str) -> str:
        return AGENCY_NAMES.get(agency, agency)

    # ── Fake / suspicious complaint detection ──────────────────────────
    def _build_freq_table(self):
        """Build a (borough, complaint_type) frequency table from
        historical data for anomaly detection."""
        if self._borough_complaint_freq is not None:
            return self._borough_complaint_freq
        hist = self.load_historical()
        if hist.empty or 'borough' not in hist.columns \
                or 'complaint_type' not in hist.columns:
            self._borough_complaint_freq = pd.DataFrame()
            return self._borough_complaint_freq
        freq = hist.groupby(['borough', 'complaint_type']).size()\
                   .reset_index(name='count')
        self._borough_complaint_freq = freq
        return freq

    def detect_fake_complaint(self, borough: str,
                              complaint_type: str) -> dict:
        """Check whether a (borough, complaint_type) pair is historically
        rare or unseen – a signal that the complaint might be fabricated.

        Returns dict with keys: is_flagged, flag_reason, risk_score (0-100).
        """
        freq = self._build_freq_table()
        if freq.empty:
            return {'is_flagged': False, 'flag_reason': '', 'risk_score': 0}

        match = freq[(freq['borough'] == borough) &
                     (freq['complaint_type'] == complaint_type)]

        if match.empty:
            return {
                'is_flagged': True,
                'flag_reason': f'"{complaint_type}" has NEVER been reported '
                               f'in {borough} in historical data.',
                'risk_score': 90
            }

        count = match['count'].values[0]
        # Total complaints in that borough
        borough_total = freq[freq['borough'] == borough]['count'].sum()
        pct = (count / borough_total * 100) if borough_total else 0

        if pct < 2:
            return {
                'is_flagged': True,
                'flag_reason': f'"{complaint_type}" is very rare in {borough} '
                               f'(only {pct:.1f}% of {borough} complaints).',
                'risk_score': 70
            }
        if pct < 5:
            return {
                'is_flagged': True,
                'flag_reason': f'"{complaint_type}" is uncommon in {borough} '
                               f'({pct:.1f}% of complaints).',
                'risk_score': 45
            }
        return {'is_flagged': False, 'flag_reason': '', 'risk_score': 0}

    # ── Save new complaint ─────────────────────────────────────────────
    def save_complaint(self, data: dict) -> str:
        """Append a new complaint row to live_complaints.csv.

        Returns the generated unique complaint ID.
        """
        complaint_id = f"NYC311-{datetime.now().strftime('%Y%m%d')}-" \
                       f"{np.random.randint(100000, 999999)}"
        data['unique_key'] = complaint_id
        data['created_date'] = datetime.now().isoformat()
        data['status'] = 'Open'

        # Determine agency automatically
        ctype = data.get('complaint_type', '')
        agency = self.get_agency_for_type(ctype)
        data['agency'] = agency
        data['agency_name'] = self.get_agency_name(agency)

        # Ensure all expected columns exist
        live = self.load_live()
        if live.empty:
            cols = pd.DataFrame(columns=[
                'unique_key', 'created_date', 'closed_date', 'agency',
                'agency_name', 'complaint_type', 'descriptor', 'descriptor_2',
                'location_type', 'incident_zip', 'incident_address',
                'street_name', 'borough', 'latitude', 'longitude', 'status',
                'citizen_name', 'citizen_mobile', 'is_flagged', 'flag_reason',
                'source'
            ])
            cols.to_csv(self.live_path, index=False)

        row = pd.DataFrame([data])
        row.to_csv(self.live_path, mode='a', header=False, index=False)

        # Invalidate cache so next get_combined() re-reads
        self.reload()
        return complaint_id