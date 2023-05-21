import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.tree import export_graphviz
import graphviz
%matplotlib inline

# The dataset is being read and stored into the variable 'data'
data = pd.read_csv("London_bike_data.csv")

# The head function prints the first 5 rows of the dataset. Using this, we can see the datatypes used in each column. This also makes it easier for us to understand which columns we would require for our algorithms, allowing the data cleaning process to be easier.
data.head()

# The column headings are printed for ease of viewing
data.columns

# The shape function tells us that there are 13060 rows of data and 12 columns.
data.shape

# Here we can see the data types for each column. Most of them are integers however temperature, temperature_feels, humidity and wind_speed are float values. Bike_rented is an object more specifically it is a nominal value.
data.dtypes

# This stacked bar chart shows the number of days that has bikes rented on weekends compared to weekdays. For example, 2012 weekdays had a very low number of bikes rented. The graph is divided by weekends and weekdays, while the bars are divided by the bike rented. Labels for the number of days that fit the bike_rented are added for readability.

# Group the data by is_holiday and bike_rented
grouped = data.groupby(['is_weekend', 'bike_rented']).size().unstack()

# Create a stacked bar chart
ax = grouped.plot(kind='bar', stacked=True)

# Add the labels in the center of the bar
for i in ax.containers:
    ax.bar_label(i, label_type='center', fontsize=10)

# Show the plot
plt.show()

#Creates a barplot with the x-axis representing the "bike_rented" column from the "data" DataFrame, the y-axis representing the "temperature" column, and the hue representing the "season" column. The palette is set to "ch:.25" to choose a color palette.
ax = sns.barplot(x="bike_rented", y="temperature", hue="season", data=data, palette="ch:.25")

# Move the legend to the lower right corner of the plot
ax.legend(bbox_to_anchor=(1.05, 0), loc='lower left', title='Season')

# Adjust the plot margins to make room for the legend
plt.subplots_adjust(right=0.8)

# Create a copy of the data. If the original dataset is used without copying, the values change and the algorithm doesnt run correctly.
data_copy = data.copy()

# Perform string counting on the copied dataframe
verylow = data_copy['bike_rented'].str.count('very low').sum()
print("very low: ", verylow)

low = data_copy['bike_rented'].str.count('low').sum()
print("low: ", low)

medium = data_copy['bike_rented'].str.count('medium').sum()
print("medium: ", medium)

high = data_copy['bike_rented'].str.count('high').sum()
print("high: ", high)

veryhigh = data_copy['bike_rented'].str.count('very high').sum()
print("very high: ", veryhigh)

# Define the labels for the data, as well as the values we got from the above lines of code
bike_rented = ['very low', 'low', 'medium', 'high', 'very high']
rented_counts = [2629, 5271, 2577, 5212, 2592]

# Define the colors for each class using hexadecimal codes
colors = ['#FF9999','#66B3FF','#99FF99','#FFCC99','#D6A5C9']

# Create the pie chart rented_counts is passed as the first argument, representing the values to plot. labels=bike_rented is passed to label each section of the pie chart. colors=colors is passed to specify the colors to use for each section of the pie chart. autopct='%1.1f%%' is passed to add percentage values to each section of the pie chart, rounded to one decimal place. startangle=90 is passed to rotate the starting angle of the chart by 90 degrees.
plt.pie(rented_counts, labels=bike_rented, colors=colors, autopct='%1.1f%%', startangle=90)

# Add title
plt.title("Bikes rented")

# Show the chart
plt.show()


# Data cleaning: we dropped the date and id columns from the dataset because we did not require them for our algorithm. We just required all the information about the day as well as the weather.

data = data.drop(["date", "id"], axis=1)


# LOGISTIC REGRESSION


# Logistic regression is used to estimate the probability of an event occuring. We also used k-fold cross validation to test how well our model did.

# First we split the data into the two variables: features (x) and targets (y)
X = data.drop('bike_rented', axis=1)
y = data['bike_rented']

# We then define the model we are using, in this case its logistic regression
model = LogisticRegression()

# Perform k-fold cross-validation and calculate the average accuracy score
kf = KFold(n_splits=5, shuffle=True, random_state=101)  # We split the data into 5 groups
accuracy_scores = []

# We iterate through the loop 5 times (one for each fold)
for train_index, test_index in kf.split(X):
    # Split the data into training and testing sets for the current fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the machine learning model to the training data
    model.fit(X_train, y_train)

    # Calculate and store the accuracy score of the model on the testing data
    accuracy_scores.append(model.score(X_test, y_test))

    # Display the accuracy score of the model on the testing data
    print('Accuracy Score:', model.score(X_test, y_test))

print("Average accuracy score:", np.mean(accuracy_scores))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Fit the model on the training data
model.fit(X_train, y_train)

# Use the trained model to predict on the testing data
y_pred = model.predict(X_test)

# Print the classification report on the testing data
print(classification_report(y_test, y_pred))

# Calculate precision, recall and F1-score for the testing data
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Here we can see that the model isn't the best. The accuracy for each fold is displayed and the overall accuracy is also calculated. The closer the accuracy is to 1 the better the model is preforming. The accuracy report displayed shows how well the model did in precision and recall.

# The precision shows how many TRUE POSITIVES were correctly categorised out of all the positives predicted. This means that there were more falsely categorised positives compared to true positives, lowering the score. This is calculated by the equation: TP/(FP+TP).

# The recall shows how many TRUE POSITIVES were correctly catergised out of all the positives (FALSE NEGATIVES + TRUE POSITIVES). This means that there were more falsely categorised negatives compared to true positives, lowering the score. This is calculated by the equation: TP/(FN+TP).

# The f1 score is the mean of precision and recall. Once again, this model isn't the best solution for the classification problem provided.

from sklearn.metrics import confusion_matrix

# Use the trained model to predict on the testing data
y_pred = model.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print(cm)

import matplotlib.pyplot as plt
import numpy as np

# Define the confusion matrix
confusion_matrix = np.array([[171, 141, 151, 303, 34],
                             [23, 224, 213, 33, 291],
                             [137, 196, 277, 127, 57],
                             [143, 105, 108, 362, 41],
                             [1, 75, 10, 23, 672]])

# Define the labels for the confusion matrix
labels = ['0', '1', '2', '3', '4']

# Define the colors for the confusion matrix
cmap = plt.cm.Oranges  # change this line to use the orange colormap

# Create the figure and the axis objects
fig, ax = plt.subplots()

# Create the table
table = ax.table(cellText=confusion_matrix.astype(int),
                 cellColours=cmap(confusion_matrix.astype(float) / confusion_matrix.max()),
                 cellLoc='center',
                 colLabels=labels,
                 rowLabels=labels,
                 loc='center')

# Modify the font size of the table
table.auto_set_font_size(False)
table.set_fontsize(14)

# Modify the size of the figure
fig.set_size_inches(8, 8)

# Remove the axis
ax.axis('off')

# Display the table
plt.show()


# DECISION TREES

# Decision trees models decisions and their possible consequences in a tree-like structure. The way this function works is by continuously dividing the data into smaller groups based on the most revealing features until a stopping requirement is satisfied. They are used for both classification and regression applications. We also used k-fold cross validation to test how well our model did.

# First we split the data into the two variables: features (x) and targets (y)
X = data.drop('bike_rented', axis=1)
y = data['bike_rented']

# Here we defined the model we are using
model = DecisionTreeClassifier()

# Perform k-fold cross-validation and calculate the average accuracy score
kf = KFold(n_splits=5, shuffle=True, random_state=101)
Accuracy = []

# We iterate through the loop 5 times (one for each fold)
for train_index, test_index in kf.split(X):
    # Split the data into training and testing sets for the current fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the machine learning model to the training data
    model.fit(X_train, y_train)

    # Calculate and store the accuracy score of the model on the testing data
    accuracy_scores.append(model.score(X_test, y_test))

    # Display the accuracy score of the model on the testing data
    print('Accuracy Score:', model.score(X_test, y_test))

print("Average accuracy score:", np.mean(accuracy_scores))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Fit the model on the training data
model.fit(X_train, y_train)

# Use the trained model to predict on the testing data
y_pred = model.predict(X_test)

# Print the classification report on the testing data
print(classification_report(y_test, y_pred))

# Calculate precision, recall and F1-score for the testing data
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Generate visualization of decision tree
tree_data = export_graphviz(model, out_file=None,  # Export the decision tree in DOT format
                            feature_names=X.columns,  # Specify the feature names
                            class_names=['low', 'medium', 'high', 'very low', 'very high'],  # Specify the class names
                            filled=True, rounded=True,  # Style the nodes
                            special_characters=True)  # Allow special characters for graph

# Create graph object from DOT data
graph = graphviz.Source(tree_data)

# This creates a PDF file with the decision tree. Due to the large nature of the dataset, the tree isn't readable unless you zoom into it.
graph.render('decision_tree', view=True)

# Here we can see that the model doing better than logistic regression. The accuracy for each fold is displayed and the overall accuracy is also calculated. The closer the accuracy is to 1 the better the model is preforming. The accuracy report displayed shows how well the model did in precision and recall.

# The precision shows how many TRUE POSITIVES were correctly categorised out of all the positives predicted. This means that there were a decent amount of truly categorised positives compared to false positives, raising the score. This is calculated by the equation: TP/(FP+TP).

# The recall shows how many TRUE POSITIVES were correctly catergised out of all the positives (FALSE NEGATIVES + TRUE POSITIVES). This means that there were more truly categorised positives compared to false negatives, increasing the score. This is calculated by the equation: TP/(FN+TP).

# The f1 score is the mean of precision and recall. This solution is better than the logistic regression solution.

from sklearn.metrics import confusion_matrix

# Use the trained model to predict on the testing data
y_pred = model.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print(cm)

import matplotlib.pyplot as plt
import numpy as np

# Define the confusion matrix
confusion_matrix = np.array([[565, 13, 113, 109, 0],
                             [10, 576, 124, 1, 73],
                             [118, 134, 530, 10, 2],
                             [109, 1, 16, 633, 0],
                             [0, 84, 2, 0, 695]])

# Define the labels for the confusion matrix
labels = ['0', '1', '2', '3', '4']

# Define the colors for the confusion matrix
cmap = plt.cm.Greens

# Create the figure and the axis objects
fig, ax = plt.subplots()

# Create the table
table = ax.table(cellText=confusion_matrix.astype(int),
                 cellColours=cmap(confusion_matrix.astype(float) / confusion_matrix.max()),
                 cellLoc='center',
                 colLabels=labels,
                 rowLabels=labels,
                 loc='center')

# Modify the font size of the table
table.auto_set_font_size(False)
table.set_fontsize(14)

# Modify the size of the figure
fig.set_size_inches(8, 8)

# Remove the axis
ax.axis('off')

# Display the table
plt.show()



# NEURAL NETWORKS

# Neural networks is an algorithm that models after the human brain structure and function.. It has interconnected nodes that process and communicate information and is a good predictive analysis technique.

# First we split the data into the two variables: features (x) and targets (y)
X = data.drop('bike_rented', axis=1)
y = data['bike_rented']

# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
x = scaler.fit_transform(X)

# Initialize the neural network classifier
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Fit the model on the training data
model.fit(X_train, y_train)

# Perform k-fold cross-validation and calculate the average accuracy score
kf = KFold(n_splits=5, shuffle=True, random_state=101)
accuracy_scores = []

# We iterate through the loop 5 times (one for each fold)
for train_index, test_index in kf.split(X):
    # Split the data into training and testing sets for the current fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the machine learning model to the training data
    model.fit(X_train, y_train)

    # Calculate and store the accuracy score of the model on the testing data
    accuracy_scores.append(model.score(X_test, y_test))

    # Display the accuracy score of the model on the testing data
    print('Accuracy Score:', model.score(X_test, y_test))

print("Average accuracy score:", np.mean(accuracy_scores))

# Use the trained model to predict on the testing data
y_pred = model.predict(X_test)

# Print the classification report on the testing data
print(classification_report(y_test, y_pred))

# Calculate precision, recall and F1-score for the testing data
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Here we can see that the model doing better than logistic regression and similar to decision trees. The accuracy for each fold is displayed and the overall accuracy is also calculated. The closer the accuracy is to 1 the better the model is preforming. The accuracy report displayed shows how well the model did in precision and recall.

# The precision shows how many TRUE POSITIVES were correctly categorised out of all the positives predicted. This means that there were a decent amount of truly categorised positives compared to false positives, raising the score. This is calculated by the equation: TP/(FP+TP).

# The recall shows how many TRUE POSITIVES were correctly catergised out of all the positives (FALSE NEGATIVES + TRUE POSITIVES). This means that there were more truly categorised positives compared to false negatives, increasing the score. This is calculated by the equation: TP/(FN+TP).

# The f1 score is the mean of precision and recall. So far, this is one of the best solution we have come up with.

# If you are getting this error: "no method available for opening 'decision_tree.pdf'" its because you need to have a pdf viewer installed on your device for the pdf to open automatically after the running the code

from sklearn.metrics import confusion_matrix

# Use the trained model to predict on the testing data
y_pred = model.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print(cm)

import matplotlib.pyplot as plt
import numpy as np

# Define the confusion matrix
confusion_matrix = np.array([[331, 7, 59, 111, 0],
                             [20, 285, 96, 14, 103],
                             [172, 74, 157, 86, 15],
                             [219, 3, 41, 288, 1],
                             [0, 23, 5, 7, 495]])

# Define the labels for the confusion matrix
labels = ['0', '1', '2', '3', '4']

# Define the colors for the confusion matrix
cmap = plt.cm.Blues

# Create the figure and the axis objects
fig, ax = plt.subplots()

# Create the table
table = ax.table(cellText=confusion_matrix.astype(int),
                 cellColours=plt.cm.Blues(confusion_matrix.astype(float) / confusion_matrix.max()),
                 cellLoc='center',
                 colLabels=labels,
                 rowLabels=labels,
                 loc='center')

# Modify the font size of the table
table.auto_set_font_size(False)
table.set_fontsize(14)

# Modify the size of the figure
fig.set_size_inches(8, 8)

# Remove the axis
ax.axis('off')

# Display the table
plt.show()



# RANDOM FOREST


# Random forest is one of the best predictive, analytical algorithm. It works by creating a large number of decision trees and combining their predictions to make a final prediction. The tree is built on a random subset of the data and a random subset of the features, which helps to reduce overfitting and improve generalization.

# First we split the data into the two variables: features (x) and targets (y)
X = data.drop('bike_rented', axis=1)
y = data['bike_rented']

Accuracy = []

model = RandomForestClassifier(n_estimators=100, random_state=42)

# Define k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate through the k-fold cross-validation splits
for train_index, test_index in kfold.split(X):
    # Get X and y train/test
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Fit the model on the training data
    model.fit(X_train, y_train)
    # Use the trained model to predict on the testing data
    y_pred = model.predict(X_test)
    # Calculate and print the metrics
    print('Accuracy Score:', model.score(X_test, y_test))
    Accuracy.append(model.score(X_test, y_test))

# Print the average accuracy score
print("Average accuracy score:", np.mean(Accuracy))

# Calculate and print the metrics
print(classification_report(y_test, y_pred))

# Calculate precision, recall and F1-score for the testing data
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)

# Here we can see that this model is doing the best out of all the models. The accuracy for each fold is displayed and the overall accuracy is also calculated. The closer the accuracy is to 1 the better the model is preforming. The accuracy report displayed shows how well the model did in precision and recall.

# The precision shows how many TRUE POSITIVES were correctly categorised out of all the positives predicted. This means that there were a high amount of truly categorised positives compared to false positives, raising the score. This is calculated by the equation: TP/(FP+TP).

# The recall shows how many TRUE POSITIVES were correctly catergised out of all the positives (FALSE NEGATIVES + TRUE POSITIVES). This means that there were more truly categorised positives compared to false negatives, increasing the score. This is calculated by the equation: TP/(FN+TP).

# The f1 score is the mean of precision and recall. So far, this is the best solution we have come up with.

from sklearn.metrics import confusion_matrix

# Use the trained model to predict on the testing data
y_pred = model.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print(cm)
import matplotlib.pyplot as plt
import numpy as np

# Define the confusion matrix
confusion_matrix = np.array([[398, 3, 45, 55, 0],
                             [4, 426, 69, 0, 40],
                             [72, 81, 362, 4, 0],
                             [51, 0, 2, 472, 0],
                             [0, 47, 0, 0, 481]])

# Define the labels for the confusion matrix
labels = ['0', '1', '2', '3', '4']

# Define the colors for the confusion matrix
cmap = plt.cm.Reds

# Create the figure and the axis objects
fig, ax = plt.subplots()

# Create the table
table = ax.table(cellText=confusion_matrix.astype(int),
                 cellColours=plt.cm.Reds(confusion_matrix.astype(float) / confusion_matrix.max()),
                 cellLoc='center',
                 colLabels=labels,
                 rowLabels=labels,
                 loc='center')

# Modify the font size of the table
table.auto_set_font_size(False)
table.set_fontsize(14)

# Modify the size of the figure
fig.set_size_inches(8, 8)

# Remove the axis
ax.axis('off')

# Display the table
plt.show()
