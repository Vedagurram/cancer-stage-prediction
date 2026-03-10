#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
import seaborn as sn

# Load the dataset
df = pd.read_csv('cancer_patients.csv')
df.head()

# Encode 'Level' column with numerical values
df.Level.replace('High', '2', inplace=True)
df.Level.replace('Medium', '1', inplace=True)
df.Level.replace('Low', '0', inplace=True)

# Check the 'Level' column
df.Level

# Prepare features and labels
X = df.drop(['Patient Id', 'Level'], axis=1)
y = df.Level

# Check for any missing values
print(df.isnull().sum())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-Means Clustering (used for visualization)
km = KMeans(n_clusters=3)
y_pred_kmeans = km.fit_predict(df[['Level', 'Age']])
df['Cluster'] = y_pred_kmeans

# Visualizing K-Means clusters
df1 = df[df.Cluster == 0]
df2 = df[df.Cluster == 1]
df3 = df[df.Cluster == 2]

plt.scatter(df1.Age, df1['Level'], color='yellow')
plt.scatter(df2.Age, df2['Level'], color='pink')
plt.scatter(df3.Age, df3['Level'], color='black')
plt.xlabel('Age')
plt.ylabel('Level')
plt.title('K-Means Clustering')
plt.show()

# Scaling individual features
scaler.fit(df[['Level']])
df['Level'] = scaler.transform(df[['Level']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])

scaler.fit(df[['Air Pollution']])
df['Air Pollution'] = scaler.transform(df[['Air Pollution']])

scaler.fit(df[['Gender']])
df['Gender'] = scaler.transform(df[['Gender']])

scaler.fit(df[['Alcohol use']])
df['Alcohol use'] = scaler.transform(df[['Alcohol use']])

scaler.fit(df[['Dust Allergy']])
df['Dust Allergy'] = scaler.transform(df[['Dust Allergy']])

# K-Nearest Neighbors Classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test, y_pred_knn)
print("KNN Confusion Matrix:\n", cm_knn)
print("KNN Accuracy:", knn.score(X_test, y_test))

# Support Vector Machine Classifier
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
print("SVM Accuracy:", svm_model.score(X_test, y_test))

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train, y_train)
print("Random Forest Accuracy:", rfc.score(X_test, y_test))

# Ridge Regression (used for comparison purposes)
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
print("Ridge Regression Score:", ridge_model.score(X_test, y_test))

# Multi-Layer Perceptron (MLP) Classifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
print("MLP Confusion Matrix:\n", cm_mlp)
print("MLP Accuracy:", mlp.score(X_test, y_test))

# Visualize the confusion matrices for different models
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sn.heatmap(cm_knn, annot=True, fmt='d')
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.subplot(2, 2, 2)
sn.heatmap(confusion_matrix(y_test, svm_model.predict(X_test)), annot=True, fmt='d')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.subplot(2, 2, 3)
sn.heatmap(cm_mlp, annot=True, fmt='d')
plt.title('MLP Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.subplot(2, 2, 4)
sn.heatmap(confusion_matrix(y_test, rfc.predict(X_test)), annot=True, fmt='d')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')

plt.tight_layout()
plt.show()