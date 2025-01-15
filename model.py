import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_fraud_detection_model():
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv("data/credit_card_transactions_with_fraud.csv")
    
    # Select only the required features
    features = ['Amount Spent', 'Card Limit', 'Amount Left']
    X = df[features]
    y = df['Fraud']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    print("Preprocessing data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions and evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print performance metrics
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model and scaler
    print("\nSaving model and scaler...")
    joblib.dump(model, 'models/fraud_detection_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Model and scaler saved successfully!")
    
    return model, scaler

if __name__ == "__main__":
    train_fraud_detection_model() 