import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings("ignore")
input_file = 'income_data.txt'
X = [] 
y = []
count_class1 = 0 
count_class2 = 0
max_datapoints = 25000 

with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break

        if '?' in line:
            continue

        data = line[:-1].split(', ') 

        if data[-1] == '<=50K' and count_class1 < max_datapoints:
            X.append(data)
            count_class1 += 1

        if data[-1] == '>50K' and count_class2 < max_datapoints:
            X.append(data)
            count_class2 += 1
X = np.array(X)
label_encoder = [] 
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
    if item.isdigit(): 
        X_encoded[:, i] = X[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)
def SVC(X,y):
    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
    classifier.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
    print('SVC \t\t',end=" ")
    print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")
    return(round(100*f1.mean(), 2))
svc=SVC(X,y)

def KNN(X,y):
    classifier = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=1))
    classifier.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    classifier = OneVsOneClassifier(KNeighborsClassifier(n_neighbors=1))
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
    print('K near\t',end=" ")
    print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")
    return(round(100*f1.mean(), 2))
knn=KNN(X,y)

def D_tree(X,y):
    
    classifier = OneVsOneClassifier(DecisionTreeClassifier(random_state=0,max_features=None))
    classifier.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    classifier = OneVsOneClassifier(DecisionTreeClassifier(random_state=0,max_features=None))
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
    print('Decision tree\t\t',end=" ")
    print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")
    return(round(100*f1.mean(), 2))
dt=D_tree(X,y)
    

def R_forest(X,y):
    
    classifier = OneVsOneClassifier(RandomForestClassifier(random_state=0))
    
    classifier.fit(X, y)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    classifier = OneVsOneClassifier(RandomForestClassifier(random_state=0))
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
    print('random forest\t',end=" ")
    print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")
    return(round(100*f1.mean(), 2))
rf=R_forest(X,y)

def naive(X,y):
    
    classifier = OneVsOneClassifier(GaussianNB())
    classifier.fit(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    classifier = OneVsOneClassifier(GaussianNB())
    classifier.fit(X_train, y_train)
    y_test_pred = classifier.predict(X_test)
    f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
    print('naive\t\t',end=" ")
    print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")
    return(round(100*f1.mean(), 2))
nb=naive(X,y)
left = [1, 2, 3, 4, 5] 

height = [svc,knn,dt,rf,nb] 
tick_label = ['SVC','KNN','DECISION_T','RANDOM_T','NAIVE'] 

plt.bar(left, height, tick_label=tick_label , width = 0.8) 
plt.xlabel('MODEL') 
plt.ylabel('F1 SCORE') 

plt.show() 

#https://www.analyticsindiamag.com/7-types-classification-algorithms/
