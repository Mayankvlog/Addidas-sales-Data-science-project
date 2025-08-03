# Credit Card Fraud Detection System 💳

An advanced machine learning solution that uses Recurrent Neural Networks (RNN) to detect fraudulent credit card transactions in real-time. This project combines deep learning with an interactive web interface for practical fraud detection.

## 🚀 Project Overview

### Machine Learning Component (`credit_card_fraud_detection.ipynb`)
- Advanced RNN model with LSTM layers
- SMOTE for handling imbalanced data
- Comprehensive performance metrics
- Interactive visualizations of results

### Web Interface (`app.py`)
- Real-time transaction analysis
- Interactive probability gauge
- Detailed transaction insights
- User-friendly interface

## 💻 Technical Stack

- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly, Seaborn
- **Web Framework**: Streamlit
- **Data Balancing**: SMOTE

## 🛠️ Setup & Installation

1. **Environment Setup**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

2. **Train Model**
```bash
jupyter notebook credit_card_fraud_detection.ipynb
```

3. **Launch Application**
```bash
streamlit run app.py
```

## 📊 Features

### Model Architecture
- 6 LSTM layers (128→24 units)
- Dropout layers (0.3)
- Multiple dense layers
- Binary classification output

### Analysis Capabilities
- Real-time fraud detection
- Probability scoring
- Confidence metrics
- Transaction risk assessment

### Visualization Tools
- Interactive gauge charts
- Performance metrics
- Transaction details
- Risk analysis

## 📈 Performance Metrics

- ROC-AUC Score
- Precision & Recall
- Confusion Matrix
- Loss Convergence

## 🗂️ Project Structure
```
project/
├── credit_card_fraud_detection.ipynb
├── app.py
├── requirements.txt
├── README.md
└── model/
    ├── fraud_detection_model.h5
    └── scaler.pkl
```

## ⚠️ Note

This system should be used as part of a comprehensive fraud detection strategy, complementing existing security measures and expert analysis.


