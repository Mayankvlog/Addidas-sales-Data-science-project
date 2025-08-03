import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import os

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
    }
    .stAlert {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = load_model('model/fraud_detection_model.h5')
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure to run the training notebook first to generate the model files.")
        return None, None

def predict_fraud(transaction_data, model, scaler, time_steps=3):
    """Predict if a transaction is fraudulent"""
    try:
        # Scale the transaction
        scaled_transaction = scaler.transform(transaction_data.reshape(1, -1))
        
        # Create sequence
        sequence = np.repeat(scaled_transaction, time_steps, axis=0)
        sequence = sequence.reshape(1, time_steps, -1)
        
        # Make prediction
        fraud_probability = model.predict(sequence)[0][0]
        fraud_prediction = 1 if fraud_probability > 0.5 else 0
        
        return fraud_probability, fraud_prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def create_gauge_chart(probability):
    """Create a gauge chart for fraud probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fraud Probability", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 33], 'color': 'green'},
                {'range': [33, 66], 'color': 'yellow'},
                {'range': [66, 100], 'color': 'red'}
            ],
        }
    ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def main():
    st.title("Credit Card Fraud Detection System 💳")
    st.markdown("---")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None or scaler is None:
        st.warning("⚠️ Model not loaded. Please check if model files exist in the 'model' directory.")
        return
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Transaction Analysis", "About"])
    
    with tab1:
        st.header("Transaction Analysis")
        
        # Create two columns
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Enter Transaction Details")
            
            # Create a form for input
            with st.form("transaction_form"):
                # Initialize empty transaction data
                transaction_data = []
                
                # Create input fields for V1-V28
                for i in range(28):
                    value = st.number_input(
                        f"V{i+1}",
                        value=0.0,
                        help=f"Feature V{i+1} of the transaction"
                    )
                    transaction_data.append(value)
                
                # Time and Amount
                time = st.number_input("Time (in seconds)", value=0)
                amount = st.number_input("Amount ($)", value=0.0, min_value=0.0)
                
                transaction_data.extend([time, amount])
                
                # Submit button
                submitted = st.form_submit_button("Analyze Transaction")
        
        with col2:
            if submitted:
                st.subheader("Analysis Results")
                
                with st.spinner('Analyzing transaction...'):
                    # Convert to numpy array
                    transaction_array = np.array(transaction_data)
                    
                    # Get prediction
                    probability, prediction = predict_fraud(transaction_array, model, scaler)
                    
                    if probability is not None:
                        # Display gauge chart
                        fig = create_gauge_chart(probability)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display prediction result
                        if prediction == 1:
                            st.error("🚨 Fraudulent Transaction Detected!")
                            confidence = probability * 100
                        else:
                            st.success("✅ Legitimate Transaction")
                            confidence = (1 - probability) * 100
                        
                        st.info(f"Confidence: {confidence:.2f}%")
                        
                        # Additional transaction details
                        st.subheader("Transaction Details")
                        details_col1, details_col2 = st.columns(2)
                        
                        with details_col1:
                            st.metric("Transaction Amount", f"${amount:.2f}")
                            st.metric("Transaction Time", f"{time} seconds")
                        
                        with details_col2:
                            st.metric("Risk Score", f"{probability * 100:.2f}%")
                            st.metric("Decision Threshold", "50%")
    
    with tab2:
        st.header("About the System")
        st.markdown("""
        ### Credit Card Fraud Detection System
        
        This system uses an advanced Recurrent Neural Network (RNN) with LSTM layers to detect fraudulent credit card transactions. The model has been trained on a dataset of credit card transactions, where each transaction is represented by 30 features:
        
        - V1-V28: Principal components obtained through PCA transformation
        - Time: Seconds elapsed between this transaction and the first transaction
        - Amount: Transaction amount
        
        #### Model Architecture
        - Multiple LSTM layers with dropout for sequence learning
        - Dense layers with various activation functions
        - Binary classification output (0: legitimate, 1: fraudulent)
        
        #### Performance Metrics
        The model has been trained and evaluated using:
        - ROC-AUC Score
        - Precision and Recall
        - Confusion Matrix
        
        #### Usage Guidelines
        1. Enter the transaction features in the form
        2. Click "Analyze Transaction"
        3. Review the prediction and confidence score
        4. Check additional transaction details
        
        #### Note
        This is a demonstration system and should be used in conjunction with other fraud detection methods in a production environment.
        """)
    
    # Add footer
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit and TensorFlow")

if __name__ == "__main__":
    main()
