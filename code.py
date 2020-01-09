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
df = pd.read_csv(path,header=None)
df.head()

# Overview of the data
df.info()
df.describe()


#Dividing the dataset set in train and test set and apply base logistic model
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
lr = LogisticRegression(random_state=101)
lr.fit(X_train,y_train)

# Calculate accuracy , print out the Classification report and Confusion Matrix.
print("Accuracy on test data:", lr.score(X_test,y_test))
y_pred = lr.predict(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print("=="*20)
print("Classification Report: \n",classification_report(y_test,y_pred))

# Copy df in new variable df1
df1 = df.copy()

# Remove Correlated features above 0.75 and then apply logistic model
corr_matrix = df1.drop(57,1).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
print("Columns to be dropped: ")
print(to_drop)
df1.drop(to_drop,axis=1,inplace=True)

# Split the new subset of data and fit the logistic model on training data
X = df1.iloc[:,:-1]
y = df1.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 42)
lr = LogisticRegression(random_state=101)
lr.fit(X_train,y_train)

# Calculate accuracy , print out the Classification report and Confusion Matrix for new data
print("Accuracy on test data:", lr.score(X_test,y_test))
y_pred = lr.predict(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))
print("=="*20)
print("Classification Report: \n",classification_report(y_test,y_pred))


# Apply Chi Square and fit the logistic model on train data use df dataset
nof_list=[20,25,30,35,40,50,55]
high_score=0
nof=0

for n in nof_list:
    test = SelectKBest(score_func=chi2 , k= n )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 42)
    X_train = test.fit_transform(X_train,y_train)
    X_test = test.transform(X_test)
    
    model = LogisticRegression(random_state=101)
    model.fit(X_train,y_train)
    print("For no of features=",n,", score=", model.score(X_test,y_test))
    if model.score(X_test,y_test)>high_score:
        high_score=model.score(X_test,y_test)
        nof=n 
print("High Score is:",high_score, "with features=",nof)

# Calculate accuracy , print out the Confusion Matrix 
y_pred = lr.predict(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))

# Apply Anova and fit the logistic model on train data use df dataset
nof_list=[20,25,30,35,40,50,55]
high_score=0
nof=0

for n in nof_list:
    test = SelectKBest(score_func=f_classif , k= n )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
    X_train = test.fit_transform(X_train,y_train)
    X_test = test.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train,y_train)
    print("For no of features=",n,", score=", model.score(X_test,y_test))

    if model.score(X_test,y_test)>high_score:
        high_score=model.score(X_test,y_test)
        nof=n 
print("High Score is:",high_score, "with features=",nof)

# Calculate accuracy , print out the Confusion Matrix 
y_pred = lr.predict(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))

# Apply PCA and fit the logistic model on train data use df dataset
nof_list=[20,25,30,35,40,50,55]
high_score=0
nof=0

for n in nof_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 42)
    pca = PCA(n_components=n)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    logistic = LogisticRegression(solver = 'lbfgs')
    logistic.fit(X_train, y_train)
    print("For no of features=",n,", score=", logistic.score(X_test,y_test))
    
    if logistic.score(X_test,y_test)>high_score:
        high_score=logistic.score(X_test,y_test)
        nof=n 
print("High Score is:",high_score, "with features=",nof)

# Calculate accuracy , print out the Confusion Matrix 
y_pred = lr.predict(X_test)
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred))

# Compare observed value and Predicted value
print("Prediction for 10 observation:    ",logistic.predict(X_test[0:10]))
print("Actual values for 10 observation: ",y_test[0:10].values)



