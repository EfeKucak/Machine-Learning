from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error
california_housing=fetch_california_housing()


# EXPLORING DATASET
print(california_housing.keys())
print(california_housing.data.shape)
print(california_housing.feature_names)
print(california_housing.DESCR)

X=california_housing.data
y=california_housing.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=42)


rf_reg= RandomForestRegressor(n_estimators=100, random_state=42,)

rf_reg.fit(X_train,y_train)

y_pred=rf_reg.predict(X_test)

mse=mean_squared_error(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred)  # Returns you the same unit as the target
print(mse)
print(rmse)


# 0.5 UNIT DIFFERENCE COMPARING TO REAL PRICE
