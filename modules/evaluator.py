"""
modules/evaluator.py

Model Evaluator for Research Paper Documentation.
Calculates Accuracy, Precision, Recall, F1-Score, and Confusion Matrices
for the Rule-Based AI engines against pseudo ground-truth labels.

Run from terminal: python modules/evaluator.py
"""
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to allow standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, classification_report)
except ImportError:
    print("ERROR: scikit-learn is required for evaluation.")
    print("Please run: pip install scikit-learn")
    sys.exit(1)

from modules.data_processor import DataProcessor
from modules.decision_engine import DecisionEngine

# ── Ground Truth Definitions ────────────────────────────────────────────
# In a rule-based system, we define "Actual" labels based on logical heuristics
# to simulate what a perfectly labeled dataset would look like.

URGENT_TYPES = [
    'HEAT/HOT WATER', 'ELECTRIC', 'WATER LEAK', 'Sewer',
    'Food Poisoning', 'Elevator', 'PLUMBING'
]

VALID_BOROUGHS = ['MANHATTAN', 'BROOKLYN', 'BRONX', 'QUEENS', 'STATEN ISLAND']


class ModelEvaluator:
    """Evaluates the performance of the rule-based AI engines."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy() if not df.empty else pd.DataFrame()

    # ── Ground Truth Generators ─────────────────────────────────────
    def _generate_fake_ground_truth(self) -> pd.Series:
        """Generate 'Actual' labels for fake complaints.
        Methodology: If descriptor is empty OR borough is invalid, 
        it is labeled 'Actually Fake' (1), else 'Legitimate' (0)."""
        if self.df.empty:
            return pd.Series(dtype=int)
        
        actual = []
        for _, row in self.df.iterrows():
            desc = str(row.get('descriptor', '')).strip()
            borough = str(row.get('borough', '')).strip().upper()
            
            # Rule: Empty description or invalid borough = Fake
            if desc == '' or desc == 'nan' or borough not in VALID_BOROUGHS:
                actual.append(1)  # Fake
            else:
                actual.append(0)  # Legitimate
                
        return pd.Series(actual, index=self.df.index)

    def _generate_urgency_ground_truth(self) -> pd.Series:
        """Generate 'Actual' labels for urgent complaints.
        Methodology: If complaint_type is inherently severe/urgent 
        (e.g., HEAT/HOT WATER, ELECTRIC), it is labeled 
        'Actually Urgent' (1), else 'Normal' (0)."""
        if self.df.empty or 'complaint_type' not in self.df.columns:
            return pd.Series(dtype=int)
            
        return self.df['complaint_type'].apply(
            lambda x: 1 if x in URGENT_TYPES else 0
        )

    # ── Evaluation Engines ──────────────────────────────────────────
    def evaluate_fake_detector(self) -> dict:
        """Evaluate Fake Complaint Detection engine."""
        if self.df.empty:
            return {}

        y_true = self._generate_fake_ground_truth()
        
        # AI Prediction: is_flagged == True -> 1, else 0
        y_pred = self.df.get('is_flagged', pd.Series(False, index=self.df.index))
        y_pred = y_pred.apply(lambda x: 1 if str(x).lower() == 'true' else 0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        return {
            'Accuracy': round(accuracy_score(y_true, y_pred) * 100, 2),
            'Precision': round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
            'Recall': round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
            'F1-Score': round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
            'True Positives (TP)': int(tp),
            'False Positives (FP)': int(fp),
            'True Negatives (TN)': int(tn),
            'False Negatives (FN)': int(fn),
            'Full Report': classification_report(y_true, y_pred, target_names=['Legitimate', 'Fake'], zero_division=0)
        }

    def evaluate_priority_engine(self) -> dict:
        """Evaluate Priority Scoring engine."""
        if self.df.empty or 'priority_score' not in self.df.columns:
            return {}

        y_true = self._generate_urgency_ground_truth()
        
        # AI Prediction: priority_score >= 55 (High/Critical) -> 1, else 0
        y_pred = self.df['priority_score'].apply(lambda x: 1 if x >= 55 else 0)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        return {
            'Accuracy': round(accuracy_score(y_true, y_pred) * 100, 2),
            'Precision': round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
            'Recall': round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
            'F1-Score': round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
            'True Positives (TP)': int(tp),
            'False Positives (FP)': int(fp),
            'True Negatives (TN)': int(tn),
            'False Negatives (FN)': int(fn),
            'Full Report': classification_report(y_true, y_pred, target_names=['Normal', 'Urgent'], zero_division=0)
        }


# ── Standalone Execution for Research Paper ─────────────────────────────
if __name__ == '__main__':
    print("="*60)
    print("AI MODEL EVALUATION METRICS FOR RESEARCH PAPER")
    print("="*60)
    print("\nLoading and processing data...")
    
    # 1. Load Data
    dp = DataProcessor()
    df = dp.get_combined()
    
    if df.empty:
        print("ERROR: No data found. Ensure nyc_dataset.csv and live_complaints.csv exist.")
        sys.exit(1)
        
    # 2. Score Data (Required for Priority Engine Evaluation)
    print("Calculating Priority Scores...")
    engine = DecisionEngine(df)
    scored_df = engine.score_all()
    
    # 3. Evaluate
    evaluator = ModelEvaluator(scored_df)
    
    # --- Fake Detector ---
    print("\n" + "="*60)
    print("1. FAKE COMPLAINT DETECTION ENGINE")
    print("="*60)
    print("Methodology: Actual Fake = Missing Descriptor OR Invalid Borough")
    print("AI Prediction: is_flagged == True\n")
    
    fake_metrics = evaluator.evaluate_fake_detector()
    if fake_metrics:
        print(f"Accuracy:  {fake_metrics['Accuracy']}%")
        print(f"Precision: {fake_metrics['Precision']}%")
        print(f"Recall:    {fake_metrics['Recall']}%")
        print(f"F1-Score:  {fake_metrics['F1-Score']}%\n")
        
        print("Confusion Matrix Counts:")
        print(f"  True Positives (Correctly flagged fake):  {fake_metrics['True Positives (TP)']}")
        print(f"  False Positives (Legit flagged as fake):   {fake_metrics['False Positives (FP)']}")
        print(f"  True Negatives (Correctly marked legit):   {fake_metrics['True Negatives (TN)']}")
        print(f"  False Negatives (Fake marked as legit):    {fake_metrics['False Negatives (FN)']}\n")
        
        print("Detailed Classification Report:")
        print(fake_metrics['Full Report'])
    else:
        print("Not enough data to evaluate.")

    # --- Priority Engine ---
    print("\n" + "="*60)
    print("2. PRIORITY SCORING ENGINE")
    print("="*60)
    print("Methodology: Actual Urgent = Complaint type in URGENT_TYPES list")
    print("AI Prediction: priority_score >= 55 (High/Critical)\n")
    
    priority_metrics = evaluator.evaluate_priority_engine()
    if priority_metrics:
        print(f"Accuracy:  {priority_metrics['Accuracy']}%")
        print(f"Precision: {priority_metrics['Precision']}%")
        print(f"Recall:    {priority_metrics['Recall']}%")
        print(f"F1-Score:  {priority_metrics['F1-Score']}%\n")
        
        print("Confusion Matrix Counts:")
        print(f"  True Positives (Correctly flagged urgent): {priority_metrics['True Positives (TP)']}")
        print(f"  False Positives (Normal flagged urgent):    {priority_metrics['False Positives (FP)']}")
        print(f"  True Negatives (Correctly marked normal):   {priority_metrics['True Negatives (TN)']}")
        print(f"  False Negatives (Urgent marked normal):     {priority_metrics['False Negatives (FN)']}\n")
        
        print("Detailed Classification Report:")
        print(priority_metrics['Full Report'])
    else:
        print("Not enough data to evaluate.")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE. Copy these metrics for your paper.")
    print("="*60)
