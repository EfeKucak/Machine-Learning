from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
oli=fetch_olivetti_faces()

print(oli.keys())
print(oli.data.shape)  # 400 data , 4096 features 
print(np.unique(oli.target))
print(oli.images.shape)
# 2D IMAGE(64x64)  >  1D(4096)

plt.figure()

for i in range(2):
    plt.subplot(1,2,i+1)
    plt.imshow(oli.images[i], cmap="gray")
    plt.axis("off")
plt.show()


X=oli.data
y=oli.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20, random_state=42)

rf_clf=RandomForestClassifier(n_estimators=50,random_state=42) # NUMBER OF ESTIMATOR(TREE NUMBERS)

rf_clf.fit(X_train,y_train)

y_pred=rf_clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

print(accuracy)