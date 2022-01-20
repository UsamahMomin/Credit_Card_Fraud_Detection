# Credit Card Fraud Detection
# Dataset: CreditCard
# Logistic Regression



# Importing Libraries
import numpy as np  # Use for arrays
import pandas as pd # Use for DataFrames
from sklearn.model_selection import train_test_split # Split the data into training and testing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score  # Check performance of model


# Read the file
path = "E:/Kaggle/Credit Card Fraud Detection/Dataset/creditcard.csv"
credit_card = pd.read_csv(path)


# First 5 rows of dataset
credit_card.head()


# Last 5 rows of dataset
credit_card.tail()


# Dataset Information
credit_card.info()


# Dimension of dataset
credit_card.shape


# Checking the number of missing values
credit_card.isnull().sum()  # We do not have any missing values


# 0 Represents Legit Transaction
# 1 Represents Fraudulent Transaction
# Distribution of Legit Transactions & Fraudulent Transactions
credit_card['Class'].value_counts() # 0 - 284315, 1 - 492


# This dataset is highly unbalanced
# 0 --> Normal Transaction
# 1 --> Fraudulent Transaction


# Seperating the data for analysis
legit = credit_card[credit_card.Class == 0]
fraud = credit_card[credit_card.Class == 1]

print(legit.shape)
print(fraud.shape)


# Statistical measures of the data
legit.Amount.describe()

fraud.Amount.describe()


# Compare the values for both transactions
credit_card.groupby('Class').mean()



# --------------------------------------------------
# Under Sampling
#---------------------------------------------------

# Build a sample dataset containing similar distribution of Normal(legit) Transaction and Fraudulent Transaction

# Number of Fraudulent Transactions ---> 492

legit_sample = legit.sample(n = 492) 


# Concatenating two DataFrames
new_dataset = pd.concat([legit_sample, fraud], axis=0)

new_dataset.head()
new_dataset.tail()

new_dataset['Class'].value_counts() # 0--> 492, 1--> 492


# Compare the values for both transactions of new dataset
new_dataset.groupby('Class').mean()


# Splitting the data into Features and Targets
x = new_dataset.drop(columns = 'Class', axis=1)
y = new_dataset['Class']

print(x)

print(y)


# Split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

print(x.shape, x_train.shape, x_test.shape)


# Model Training
# Logistic Regression
model = LogisticRegression()


# Training the Logistic Regression Model with Training Data
model.fit(x_train, y_train)


# Model Evaluation
# Accuracy Score

# Accuracy on Training Data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print("Accuracy on Training data : " , training_data_accuracy)


# Accuracy on Test Data

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print("Accuracy score on Test data : " , test_data_accuracy)


# The accuracy score is very similar of both training data and testing data, so our model performed really well