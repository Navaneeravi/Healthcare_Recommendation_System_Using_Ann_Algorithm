import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import scikitplot as skplt
import keras


#Read dataset
dataset = pd.read_csv('diabetes.csv')
print(dataset.head())
print(dataset.info())
print(dataset.describe())

#pre-processing
print("Checking NULL values:",dataset.isnull().any())

#EDA
"""Data visualization"""

"""heat map"""
corr = dataset.corr(method='pearson')
mask = np.triu(np.ones_like(corr, dtype=np.bool)) 
fig = plt.subplots(figsize=(25, 15))
sns.heatmap(dataset.corr(), annot=True,fmt='.2f',mask=mask)
plt.show()

"""Box plot"""
data1=dataset.drop('Outcome',axis=1)
data1.plot(kind='box', subplots=True, layout=(4,4), sharex=False,sharey=False ,figsize =(15,15))
plt.show()

"""pie graph"""
dataset['Outcome'].value_counts().plot(kind='pie',colors=['Brown', 'Green'],autopct='%1.1f%%',figsize=(9,9))
plt.show()

#model selection
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =9)


'''RANDOM FOREST'''

#Create a Gaussian Classifier
rf_clf=RandomForestClassifier(n_estimators=100)
rf_clf.fit(x_train,y_train)

rf_ypred=rf_clf.predict(x_test)
print('\n')
print("------Accuracy------")
rf=accuracy_score(y_test, rf_ypred)*100
RF=('RANDOM FOREST Accuracy:',accuracy_score(y_test, rf_ypred)*100,'%')
print(RF)
print('\n')
print("------Classification Report------")
print(classification_report(rf_ypred,y_test))
print('\n')
print('Confusion_matrix')
rf_cm = confusion_matrix(y_test, rf_ypred)
print(rf_cm)
print('\n')
tn = rf_cm[0][0]
fp = rf_cm[0][1]
fn = rf_cm[1][0]
tp = rf_cm[1][1]
Total_TP_FP=rf_cm[0][0]+rf_cm[0][1]
Total_FN_TN=rf_cm[1][0]+rf_cm[1][1]
specificity = tn / (tn+fp)
rf_specificity=format(specificity,'.3f')
print('RF_specificity:',rf_specificity)
print()

# plt.figure()
# skplt.estimators.plot_learning_curve(RandomForestClassifier(), x_train, y_train,
#                                      cv=7, shuffle=True, scoring="accuracy",
#                                      n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
#                                      title="Random Forest Digits Classification Learning Curve");

plt.figure()                                   
sns.heatmap(confusion_matrix(y_test,rf_ypred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()




'''ANN'''
print("Artificial Neural Network")
print()
ann = keras.models.Sequential()

ann.add(keras.layers.Dense(units=7, activation='relu'))
ann.add(keras.layers.Dense(units=132, activation='relu'))
ann.add(keras.layers.Dense(units=279, activation='relu'))
ann.add(keras.layers.Dense(units=423, activation='relu'))
ann.add(keras.layers.Dense(units=579, activation='relu'))
ann.add(keras.layers.Dense(units=456, activation='relu'))
ann.add(keras.layers.Dense(units=303, activation='relu'))
ann.add(keras.layers.Dense(units=154, activation='relu'))
ann.add(keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=[
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
             ])
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score,fbeta_score, f1_score

n_split=10

for train_index,test_index in KFold(n_split).split(x_train):
  x_train,x_test=X[train_index],X[test_index]
  y_train,y_test=y[train_index],y[test_index]

  sc = StandardScaler()
  x_train = sc.fit_transform(x_train)
  x_test = sc.transform(x_test)
  
  
  ann.fit(x_train, y_train, epochs=20)
  y_pred = ann.predict(x_test)
  y_pred = np.array([0  if i<0.5 else 1 for i in y_pred])

  # ann.evaluate(x_test, y_test)

  print(accuracy_score(y_test,y_pred))
  print(precision_score(y_test, y_pred))
  print(fbeta_score(y_test, y_pred, beta=0.5))
  print(f1_score(y_test, y_pred))

from sklearn.metrics import accuracy_score, precision_score,fbeta_score, f1_score
print(f'accuracy    : {accuracy_score(y_test,y_pred)}')
print(f'precision   : {precision_score(y_test,y_pred)}')
print(f'fBeta score : {fbeta_score(y_test,y_pred, beta=0.5)}')
print(f'f1 score    : {f1_score(y_test,y_pred)}')

trainScore = ann.evaluate(x_test, y_test, verbose=1)
ann_=trainScore[1]*100
print('ANN Accuracy:',trainScore[1]*100,'%')
print('\n')

print('Confusion_matrix')
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('\n')
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
Total_TP_FP=cm[0][0]+cm[0][1]
Total_FN_TN=cm[1][0]+cm[1][1]
specificity = TN / (TN+FP)
ann_specificity=format(specificity)
print('ANN_specificity:',ann_specificity)
print('\n')

sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


#comparision
vals=[rf, ann_]
inds=range(len(vals))
labels=["RF", "ANN"]
fig,ax = plt.subplots()
rects = ax.bar(inds, vals)
ax.set_xticks([ind for ind in inds])
ax.set_xticklabels(labels)
plt.show()


inp=[97,66,15,140,23.2,0.487,22,0]
predd=rf_clf.predict(inp)


pre=ann.predict(inp)
if pre==1:
     print('Patient')
     print('gvg')
else:
     print('Normal')
#if predd==1:
    # print('Patient')
 #else:
    # print('Normal')