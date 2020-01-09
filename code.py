# --------------
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report,confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Load the data
#Loading the Spam data from the path variable for the mini challenge
#Target variable is the 57 column i.e spam, non-spam classes 


# Overview of the data
df=pd.read_csv(path,header=None)
#df.info()
#df.describe()

#Dividing the dataset set in train and test set and apply base logistic model
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
#print(X_train.shape,y_train.shape)

model_lr=LogisticRegression()
model_lr.fit(X_train,y_train)
y_pred=model_lr.predict(X_test)
#print(y_pred)
# Calculate accuracy , print out the Classification report and Confusion Matrix.
Accuracy=model_lr.score(X_test,y_test)

#print(classification_report(y_test,y_pred))
#print(confusion_matrix(y_test,y_pred))
# Copy df in new variable df1
df1=df.copy()
correlation_matrix=df1.drop(57,1).corr()
upper=correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(np.bool))
#print(correlation_matrix)
#print(upper)
# Remove Correlated features above 0.75 and then apply logistic model
columns_to_drop=[column for column in upper.columns if any(upper[column]>0.75)]
#print(columns_to_drop)


# Split the xnew subset of data and fit the logistic model on training data
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

# Calculate accuracy , print out the Classification report and Confusion Matrix for new data
model_lr=LogisticRegression()
model_lr.fit(X_train,y_train)
y_pred=model_lr.predict(X_test)
Accuracy=model_lr.score(X_test,y_test)

#print(classification_report(y_test,y_pred))
#print(confusion_matrix(y_test,y_pred))

# Apply Chi Square and fit the logistic model on train data use df dataset
n_features=10
test=SelectKBest(score_func=chi2,k=n_features)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
X_train_transformed=test.fit_transform(X_train,y_train)
X_test_transformed=test.transform(X_test)

lr_model3=LogisticRegression()
lr_model3.fit(X_train_transformed,y_train)

y_pred=lr_model3.predict(X_test_transformed)
# Calculate accuracy , print out the Confusion Matrix 
Accuracy=lr_model3.score(X_test_transformed,y_test)
print(Accuracy)
#print(classification_report(y_test,y_pred))
#print(confusion_matrix(y_test,y_pred))

# Apply Anova and fit the logistic model on train data use df dataset



# Calculate accuracy , print out the Confusion Matrix 


# Apply PCA and fit the logistic model on train data use df dataset

   

# Calculate accuracy , print out the Confusion Matrix 


# Compare observed value and Predicted value




