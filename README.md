# IRIS-TASK-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
df = pd.read_csv(r'C:\Users\I SEvEN\Desktop\CodSoft tasks\dataset\data tasks\IRIS.csv')
df.head()
iris = load_iris()

# transform the iris data to a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# add the 'species' column to the DataFrame
df['species'] = iris.target

# replace the species codes with the species names
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# show the first 5 rows of the DataFrame
print(df.head())

sns.pairplot(df, hue='species')
plt.show()
from sklearn.model_selection import train_test_split


X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df.head()
from sklearn.linear_model import LogisticRegression

# Initialize LogisticRegression
model = LogisticRegression()

# Fit the model with training data
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(new_sample)
print(f'Prediction: {prediction[0]}')
