from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# DATASET

cancer=load_breast_cancer()

# CREATE DF FROM CANCER BUNCH FILE

df=pd.DataFrame(data=cancer.data, columns=cancer.feature_names)


"""if "Target" in df.columns:
     print("There is a target column")

else:
    print("There is no target column")"""


df["Target"]=cancer.target


# FEATURE SELECTION AND PREPROCESSING

X=cancer.data
y=cancer.target



# TRAIN TEST SPLIT

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# FEATURE SCALING

scale=StandardScaler()

scale.fit_transform(X_train)  # FIT AND APPLY
scale.transform(X_test) # APPLY THE SAME PROCESS- 

# MODEL SELECTION

knn=KNeighborsClassifier(n_neighbors=3)  # 3 Neighbors will be considered.

knn.fit(X_train,y_train)

# EVALUATE THE RESULT

y_pred=knn.predict(X_test)


# ACCURACY RESULT

accuracy=accuracy_score(y_test,y_pred)

print(accuracy)

c_matrix=confusion_matrix(y_test,y_pred)


# HYPERPARAMETER CONTROL
accuracies=[]
k_values=[]
for k in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)
    accuracies.append(accuracy)
    k_values.append(k)

best_index = accuracies.index(max(accuracies))  # max accuracy index
best_k = k_values[best_index]  # Getting K_value for the max index
best_accuracy = max(accuracies)  

print(f"Best K value is : {best_k} and Best Accuracy is : {best_accuracy} ")


plt.figure(figsize=(12,6))

plt.plot(k_values,accuracies,marker="o")

plt.xlabel("K_Values")
plt.ylabel("Accuracy Values")
plt.xticks(k_values)
plt.grid(True)
plt.show()

