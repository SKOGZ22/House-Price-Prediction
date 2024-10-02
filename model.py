from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def train_linear_model(X_train, y_train):
    # Create and train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest_model(X_train, y_train):
    # Create and train a random forest model
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    return rf_model
