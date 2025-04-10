from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

import warnings

#DATASET ANALYSIS

iris=load_iris()


print(iris.keys())
print(iris.feature_names)
print(iris.data.shape)
print(iris.target)
print(iris.target_names)


X=iris.data  #FEATURES
y=iris.target  # TARGET

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# MODEL DEFINING

tree_clf=DecisionTreeClassifier(criterion="gini", max_depth=6,random_state=42) # or entropi # HYPERPARAMETERS

tree_clf.fit(X_train,y_train)

# TREE EVALUATION

y_pred=tree_clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)

print(f"MOdel accuracy score : {accuracy}")

confusion_matrix=confusion_matrix(y_test,y_pred)

# VISUALIZATION OF TREE : plot_tree

plt.figure(figsize=(15,10))

plot_tree(tree_clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)

plt.show()

# OBSERVATION OF FEATURES 
feature_importances=tree_clf.feature_importances_  #IT RETURNS ARRAY 

feature_names=iris.feature_names

feature_importances_sorted=sorted(zip(feature_importances,feature_names))

for importance,feature_name in feature_importances_sorted:
    print(f"feature name: {feature_name},importance: {importance}")


# FEATURE COMPARISION PART :

# %%

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import warnings
warnings.filterwarnings("ignore")

iris=load_iris()
n_classes=len(iris.target_names)
plot_colors="ryb"
featuress=iris.feature_names

for pairidx,pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
    X=iris.data[:,pair]
    y=iris.target

    clf=DecisionTreeClassifier().fit(X,y)

    ax=plt.subplot(2,3,pairidx+1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    DecisionBoundaryDisplay.from_estimator(clf,
                                            X,
                                            cmap=plt.cm.RdYlBu,
                                            response_method="predict",
                                            ax=ax,
                                            xlabel=iris.feature_names[pair[0]],
                                            ylabel=iris.feature_names[pair[1]])
    
    for i,color in zip(range(n_classes),plot_colors):
        idx=np.where(y==i)
        plt.scatter(X[idx,0],X[idx,1],c=color, label=iris.target_names[i],cmap=plt.cm.RdYlBu,edgecolors="black")

plt.legend()
