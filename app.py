import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('models/fraud_detection_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract only the required features
        features = {
            'Amount Spent': float(data['amount_spent']),
            'Card Limit': float(data['card_limit']),
            'Amount Left': float(data['amount_left'])
        }
        
        # Create DataFrame with the features
        df = pd.DataFrame([features])
        
        # Scale the features
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        # Return prediction result
        return jsonify({
            'fraud_detected': bool(prediction),
            'message': 'Transaction is fraudulent' if prediction == 1 else 'Transaction is legitimate'
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

if __name__ == '__main__':
    app.run(debug=True)
