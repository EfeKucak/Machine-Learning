from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error ,root_mean_squared_error
diabetes=load_diabetes()

#print(diabetes.feature_names)
#print(diabetes.target)


X=diabetes.data
y=diabetes.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#DECISION TREE REGRESSION

tree_reg=DecisionTreeRegressor(random_state=42)

tree_reg.fit(X_train,y_train)

# PREDICTION

y_pred=tree_reg.predict(X_test)

# MSE - RMNSE 

mse=mean_squared_error(y_test,y_pred)

print(f"Mse: {mse}")

rmse=root_mean_squared_error(y_test,y_pred)
print(rmse)



# %%

from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

X=np.sort(5*np.random.rand(80,1), axis=0)
y=np.sin(X).ravel()     # REQUIRES 1D ARRAY  USING RAVEL()

y[::5] += 5* (0.5- np.random.rand(16))

"""plt.scatter(X,y)
plt.xticks(X[::10].ravel())  # her 10. değeri koy

plt.show() """

reg_1=DecisionTreeRegressor(max_depth=2)        #UNDERFTTING  
reg_2=DecisionTreeRegressor(max_depth=8)        #OVERFITTING 

reg_1.fit(X,y)
reg_2.fit(X,y)

X_test=np.arange(0,5,0.05).reshape(-1,1)  # 2D ARRAY 
y_pred1=reg_1.predict(X_test)

X_test=np.arange(0,5,0.05).reshape(-1,1)
y_pred2=reg_2.predict(X_test)

plt.figure()

plt.scatter(X,y, c="red", label="data")
plt.plot(X,y, c="red", label="data")

plt.plot(X_test,y_pred1, c="blue", label="Max Depth : 2")
plt.plot(X_test,y_pred2, c="green", label="Max Depth : 5")

plt.legend()

plt.show()
            
