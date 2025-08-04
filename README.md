# Credit Card Fraud Detection System

## Project Overview
This project implements an advanced deep learning solution for detecting fraudulent credit card transactions. It combines the power of Recurrent Neural Networks (RNN) with multiple activation functions and a sophisticated architecture to achieve high-accuracy fraud detection.

## Features
- Advanced RNN architecture with 6 hidden LSTM layers
- Multiple activation functions (ReLU, tanh, sigmoid, PReLU, ELU, SELU)
- Interactive web interface using Streamlit
- Support for both single and batch predictions
- Comprehensive visualization of results
- Detailed performance metrics and analysis

## Technical Architecture
### Deep Learning Model
- **Input Layer**: LSTM with 128 units
- **Hidden Layers**:
  1. LSTM (64 units) with ReLU activation
  2. LSTM (32 units) with tanh activation
  3. LSTM (16 units) with sigmoid activation
  4. LSTM (8 units) with PReLU activation
  5. LSTM (4 units) with ELU activation
  6. LSTM (2 units) with SELU activation
- **Output Layer**: Dense layer with sigmoid activation
- **Regularization**: Dropout layers (0.2) after each LSTM layer

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Mayankvlog/Credit-card-fraud-detection-Data-science-project.git
cd Credit-card-fraud-detection-Data-science-project
```

2. Create and activate virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # Linux/Mac
myenv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Running the Web Application
```bash
streamlit run app.py
```

### Features of the Web Interface
1. **Single Prediction**:
   - Input transaction details manually
   - Get instant fraud probability predictions
   - Visual indicators for fraudulent/legitimate transactions

2. **Batch Prediction**:
   - Upload CSV files with multiple transactions
   - Get predictions for all transactions
   - Download results in CSV format
   - Visualize prediction distributions
   - View confusion matrix and performance metrics

3. **Model Information**:
   - View model architecture
   - Access performance metrics
   - Read usage instructions

## Data Format
The model expects the following features:
- Time: Seconds elapsed between transactions
- V1-V28: Principal components from PCA
- Amount: Transaction amount

## Model Performance
- Uses binary cross-entropy loss
- Implements early stopping to prevent overfitting
- Includes comprehensive evaluation metrics:
  - ROC curve and AUC score
  - Precision-Recall curve
  - Confusion matrix
  - Classification report

## Project Structure
```
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies for model
├── requirements_app.txt     # Dependencies for web app
├── data/
│   └── creditcard.csv      # Dataset file
├── model/
│   ├── credit_card_fraud_model.h5  # Saved model
│   └── scaler.pkl          # Saved scaler
└── credit_card_fraud_detection.ipynb  # Model development notebook
```

## Contributing
Contributions are welcome! Please feel free to submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset: Credit Card Fraud Detection dataset
- Libraries: TensorFlow, Keras, Streamlit, scikit-learn
- Visualization: Matplotlib, Seaborn

## Contact
- Author: Mayank
- GitHub: [@Mayankvlog](https://github.com/Mayankvlog)

