# AI-Based Government Decision Intelligence System

## Overview

The AI-Based Government Decision Intelligence System is an AI-powered decision support platform developed to help government authorities efficiently manage and resolve citizen complaints. The system combines Natural Language Processing (NLP), Decision Intelligence, Statistical Analytics, Resource Optimization, and Policy Simulation to transform complaint data into actionable insights for smarter governance.

The platform provides two separate applications:

- Citizen Portal for complaint registration
- Officer Dashboard for monitoring, analysis, and decision support

---

## Features

### Citizen Portal
- Submit complaints through an interactive interface
- Select complaint location using an interactive map
- Automatic complaint ID generation
- Automatic department assignment
- Suspicious/Fake complaint detection

### NLP Engine
- Sentiment Analysis
- Urgency Detection
- Keyword Extraction
- Complaint Text Analysis

### Decision Recommendation Engine
- Multi-factor priority scoring
- Automatic complaint prioritization
- Decision recommendations
- Department assignment

### Analytics Dashboard
- Complaint trend analysis
- Borough-wise analysis
- Agency-wise analysis
- Complaint type distribution
- Status distribution
- Resolution statistics

### Crisis Detection
- Detects abnormal complaint spikes
- Early warning alerts
- Statistical anomaly detection

### Prediction
- Predicts complaint volume for the next seven days
- Confidence estimation

### Resource Optimization
- Department workload analysis
- Resource allocation optimization
- Resource reallocation suggestions

### Policy Simulator
- Budget allocation simulation
- Policy impact analysis
- Resolution rate prediction
- Response time estimation

### AI Copilot
- Natural language query interface
- Complaint statistics
- Trend analysis
- Decision recommendations

### Geographic Visualization
- Interactive complaint map
- Borough filtering
- Agency filtering
- Complaint type filtering

### PDF Report Generation
- Executive Summary
- Complaint Statistics
- AI Recommendations
- Resource Allocation
- Crisis Alerts
- Weekly Decision Intelligence Report

---

## AI Components

The system incorporates the following AI and intelligent decision-making techniques:

- Natural Language Processing (NLP)
- Sentiment Analysis
- Urgency Detection
- Keyword Extraction
- Rule-Based Decision Intelligence
- Statistical Analytics
- Resource Optimization
- Policy Simulation

---

## Supported Government Departments

- NYPD (Police Department)
- HPD (Housing Preservation and Development)
- DSNY (Department of Sanitation)
- DOT (Department of Transportation)
- DHS (Department of Homeless Services)
- DOB (Department of Buildings)
- DOHMH (Department of Health and Mental Hygiene)
- DEP (Department of Environmental Protection)
- TLC (Taxi and Limousine Commission)

---

## Technologies Used

### Programming Language
- Python

### Framework
- Streamlit

### Libraries
- Pandas
- NumPy
- Plotly
- Folium
- Streamlit-Folium
- TextBlob
- ReportLab
- Scikit-learn

---

## Project Structure

```
AI-Based-Government-Decision-Support-System
│
├── data
│   ├── nyc_dataset.csv
│   └── live_complaints.csv
│
├── modules
│   ├── analytics_engine.py
│   ├── copilot.py
│   ├── data_processor.py
│   ├── decision_engine.py
│   ├── evaluator.py
│   ├── nlp_engine.py
│   ├── optimizer.py
│   ├── simulator.py
│   └── __init__.py
│
├── citizen_app.py
├── officer_app.py
├── create_live_dataset.py
├── generate_report.py
└── README.md
```

---

## System Workflow

```
Citizen Complaint
        │
        ▼
Citizen Portal
        │
        ▼
Data Processing
        │
        ▼
Fake Complaint Detection
        │
        ▼
NLP Analysis
(Sentiment, Urgency, Keywords)
        │
        ▼
Decision Engine
(Priority Scoring)
        │
        ▼
Analytics Engine
        │
        ├── Trend Analysis
        ├── Crisis Detection
        ├── Prediction
        ├── Resource Optimization
        └── Policy Simulation
                │
                ▼
Officer Dashboard
        │
        ▼
PDF Reports and Decision Support
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/AI-Based-Government-Decision-Support-System.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Initialize Dataset

```bash
python create_live_dataset.py
```

### Run Citizen Portal

```bash
streamlit run citizen_app.py
```

### Run Officer Dashboard

```bash
streamlit run officer_app.py
```

---

## Future Enhancements

- Machine Learning-based complaint classification
- Deep Learning-based sentiment analysis
- Multilingual complaint support
- Voice-based complaint submission
- Real-time GIS integration
- Explainable AI (XAI)
- Large Language Model (LLM) integration
- Cloud deployment

---

## Author

**Muiz Takey**
**Owais Batte**
**Rahil Mazgaonkar**
Bachelor of Engineering (Artificial Intelligence and Machine Learning)

---

## License

This project is developed for academic and research purposes. It demonstrates the application of Artificial Intelligence, Natural Language Processing, Decision Intelligence, and Data Analytics for improving government complaint management and supporting data-driven governance.
