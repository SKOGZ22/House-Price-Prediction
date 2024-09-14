import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
housing=fetch_california_housing()
df=pd.DataFrame(housing.data,columns=housing.feature_names)
df['PRICE']=housing.target
X=df.drop('PRICE',axis=1)
y=df['PRICE']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
print(f"Mean Squared Error : {mse}")
import numpy as np
rmse=np.sqrt(mse)
print(f"Root Mean Squared Error:{rmse}")
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print(f"R^2 Score:{r2}")
import matplotlib.pyplot as plt
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
residuals= y_test - y_pred
plt.scatter(y_pred,residuals)
plt.axhline(y=0,color='r',linestyle='--')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Prices")
plt.show()
coefficient=model.coef_
feature_importance=pd.DataFrame(coefficient,X.columns,columns=["Importance"])
print(feature_importance)
from sklearn.ensemble import RandomForestRegressor
rf_model=RandomForestRegressor()
rf_model.fit(X_train,y_train)
rf_pred=rf_model.predict(X_test)
rf_mse=mean_squared_error(y_test,rf_pred)
print(f"Random Forest MSE: {rf_mse}")
import joblib
joblib.dump(model,'house_price_model.pk1')