"""
Initialize the live_complaints.csv file with the correct header schema.
Run this once before launching either Streamlit app.
"""
import os

import pandas as pd


def create_live_complaints_csv():
    """Create an empty live_complaints.csv that mirrors the NYC 311 schema
    plus extra citizen-facing fields."""
    headers = [
        'unique_key', 'created_date', 'closed_date', 'agency', 'agency_name',
        'complaint_type', 'descriptor', 'descriptor_2', 'location_type',
        'incident_zip', 'incident_address', 'street_name', 'cross_street_1',
        'cross_street_2', 'intersection_street_1', 'intersection_street_2',
        'address_type', 'city', 'landmark', 'facility_type', 'status',
        'due_date', 'resolution_description', 'resolution_action_updated_date',
        'community_board', 'council_district', 'police_precinct', 'bbl',
        'borough', 'x_coordinate_state_plane', 'y_coordinate_state_plane',
        'open_data_channel_type', 'park_facility_name', 'park_borough',
        'vehicle_type', 'taxi_company_borough', 'taxi_pick_up_location',
        'bridge_highway_name', 'bridge_highway_direction', 'road_ramp',
        'bridge_highway_segment', 'latitude', 'longitude', 'location',
        # ── Extra fields for citizen portal ──
        'citizen_name', 'citizen_mobile', 'is_flagged', 'flag_reason'
    ]

    os.makedirs('data', exist_ok=True)
    filepath = os.path.join('data', 'live_complaints.csv')

    if not os.path.exists(filepath):
        df = pd.DataFrame(columns=headers)
        df.to_csv(filepath, index=False)
        print(f"✅  Created {filepath}  ({len(headers)} columns)")
    else:
        print(f"ℹ️   {filepath} already exists – skipping")


if __name__ == '__main__':
    create_live_complaints_csv()