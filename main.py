from data_preprocessing import load_data, preprocess_data
from model import train_linear_model, train_random_forest_model
from evaluation import evaluate_model
import joblib

def main():
    # Load and preprocess the data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Train Linear Regression model
    linear_model = train_linear_model(X_train, y_train)
    
    # Evaluate Linear Regression model
    print("Linear Regression Evaluation:")
    evaluate_model(linear_model, X_test, y_test)

    # Train Random Forest model
    rf_model = train_random_forest_model(X_train, y_train)
    
    # Evaluate Random Forest model
    print("Random Forest Evaluation:")
    evaluate_model(rf_model, X_test, y_test)

    # Save the Linear Regression model
    joblib.dump(linear_model, 'house_price_model.pkl')

if __name__ == "__main__":
    main()
