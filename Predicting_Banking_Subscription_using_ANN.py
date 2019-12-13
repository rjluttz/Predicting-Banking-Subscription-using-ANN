#!/usr/bin/env python3
# coding: utf-8
#--------------------------------------------------------------------------------------------------------------------------------------------
# Reading the Bank Dataset
print('Reading the Bank Dataset...')

# importing pandas for reading the datasets
print('Importing pandas for reading the datasets...')
import pandas as pd

# reading the training dataset with a ';' delimiter
print('Reading the training dataset with a ; delimiter...')
bdata_train=pd.read_csv('BankData_train.csv',delimiter=';')

# reading the testing dataset with comma as delimiter
print('Reading the testing dataset with comma as delimiter...')
bdata_test=pd.read_csv('BankData_eval.csv')

print('Successfully read training and testing dataset')
c=input('Press any key to continue...')
#--------------------------------------------------------------------------------------------------------------------------------------------
# importing matplotlib for plotting the graphs
print('Let us explore the feature variables and their relationship with the tendency to subscribe a term deposit')
print('Importing matplotlib for visualizing the relationship between the feature variables and subscription tendency')
import matplotlib.pyplot as plt

pd.crosstab(bdata_train.job,bdata_train.y).plot(kind='bar')
plt.title('Subscriptions based on Job')
plt.xlabel('Job')
plt.ylabel('No of Subscriptions')
plt.show()
# we observe that the people from the management industry are approached more
print('We observe that the people from the management industry are approached more')

pd.crosstab(bdata_train.marital,bdata_train.y).plot(kind='bar')
plt.title('Subscriptions based on Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('No of Subscriptions')
plt.show()
# we observe that the married people are approached more
print('we observe that the married people are approached more')

pd.crosstab(bdata_train.education,bdata_train.y).plot(kind='bar')
plt.title('Subscriptions based on Education')
plt.xlabel('Education')
plt.ylabel('No of Subscriptions')
plt.show()
# we observe that the people with the secondary education are approached more
print('we observe that the people with the secondary education  are approached more')

pd.crosstab(bdata_train.poutcome,bdata_train.y).plot(kind='bar',stacked=True)
plt.title('Subscriptions based on Outcome of Previous Campaign')
plt.xlabel('Outcome of Previous Campaign')
plt.ylabel('No of Subscriptions')
plt.show()
# we observe that the success of the previous campaign affect the subscription tendency
print('we observe that the success of the previous campaign affect the subscription tendency')
#--------------------------------------------------------------------------------------------------------------------------------------------
print('Performing data preprocessing before we actually fit our model...')
print('Creating dummy variables...')
# creating dummy variables for the training set

# creating a list of categorical variables to be transformed into dummy variables
category=['job','marital','education','default','housing','loan','contact','month','poutcome']

# creating a training set backup
bdata_train_new = bdata_train

# creating dummy variables and joining it to the training set
for c in category:
    new_column = pd.get_dummies(bdata_train_new[c], prefix=c)
    bdata_train_dummy=bdata_train_new.join(new_column)
    bdata_train_new=bdata_train_dummy

# removing the dummy trap
dummy_drop=['job_unknown','marital_divorced','education_unknown','default_no','housing_no','loan_no','contact_unknown','month_nov','poutcome_unknown']

# removing the unwanted columns by dropping it
bdata_train_final=bdata_train_new.drop(category+dummy_drop,axis=1)

# creating training set of features
x_train=bdata_train_final.drop(['y'],axis=1)

# creating training set of output variable
y_train=pd.DataFrame(bdata_train_final['y'])

# coding yes as '1' and no as '0'
y_train[y_train=='yes']='1'
y_train[y_train=='no']='0'

# converting it into integer categorical variable
y_train.y=y_train.y.astype('int64')
y_train.y=y_train.y.astype('category')

# creating dummy variables for the testing set

# creating a testing set backup
bdata_test_new = bdata_test

# creating dummy variables and joining it to the testing set
for c in category:
    new_column = pd.get_dummies(bdata_test_new[c], prefix=c)
    bdata_test_dummy=bdata_test_new.join(new_column)
    bdata_test_new=bdata_test_dummy

# removing the dummy trap
dummy_drop_test=['job_unknown','marital_divorced','education_unknown','default_no','housing_no','loan_no','poutcome_unknown']

# removing the unwanted columns by dropping it
bdata_test_final=bdata_test_new.drop(category+dummy_drop_test,axis=1)

# creating testing set of features
x_test=bdata_test_final.drop(['y'],axis=1)

# creating testing set of output variable
y_test=pd.DataFrame(bdata_test_final['y'])

# coding yes as '1' and no as '0'
y_test[y_test=='yes']='1'
y_test[y_test=='no']='0'

# converting it into integer categorical variable
y_test.y=y_test.y.astype('int64')
y_test.y=y_test.y.astype('category')
#--------------------------------------------------------------------------------------------------------------------------------------------
# Equalizing the number of features in training set and testing set

# finding the features which are in training set but not in testing set
print('finding the features which are in training set but not in testing set...')
print([i for i in x_train.columns.values.tolist() if i not in x_test.columns.values.tolist()])

# importing numpy for creating zero matrices
import numpy as np

# creating missing feature columns with zero entries
print('creating missing feature columns with zero entries...')
job=pd.DataFrame(np.zeros(shape=(100,1)),columns=['job_self-employed'])
default=pd.DataFrame(np.zeros(shape=(100,1)),columns=['default_yes'])
month_1=pd.DataFrame(np.zeros(shape=(100,1)),columns=['month_aug'])
month_2=pd.DataFrame(np.zeros(shape=(100,1)),columns=['month_oct'])
month_3=pd.DataFrame(np.zeros(shape=(100,1)),columns=['month_sep'])

# joining the missing feature columns to the testing feature set
print('joining the missing feature columns to the testing feature set...')
x_test=x_test.join(job)
x_test=x_test.join(default)
x_test=x_test.join(month_1)
x_test=x_test.join(month_2)
x_test=x_test.join(month_3)
#--------------------------------------------------------------------------------------------------------------------------------------------
# Standardizing the training and testing feature set
print('Standardizing the training and testing feature set...')
# importing the Standard Scaler from sklearn
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print('Data preprocessing completed successfully')
c=input('Press any key to continue...')
#--------------------------------------------------------------------------------------------------------------------------------------------
# Creating the ANN model
print('Creating the Artificial Neural Network Model...')

# importing keras library
print('Importing the keras library')
import keras
from keras.models import Sequential
from keras.layers import Dense

# initializing the ANN
print('Initializing the ANN Classifier...')
classifier = Sequential()

# Adding the input layer and the hidden layer
print('Adding the input layer and the hidden layer...')
classifier.add(Dense(units = 18 , kernel_initializer = 'uniform', activation = 'relu', input_dim = 35))

# Adding the output layer
print('Adding the output layer...')
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
print('Compiling ANN model...')
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
print('Fitting the ANN model to the training set...')
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)
#--------------------------------------------------------------------------------------------------------------------------------------------
# Plotting the ANN model
print('Plotting the Artifical Neural Network...')

# importing ann_viz from ann_visualizer
from ann_visualizer.visualize import ann_viz

ann_viz(classifier, title = 'ANN model predicting Banking Subscriptions')
#--------------------------------------------------------------------------------------------------------------------------------------------
# Evaluating the performance of the ANN model
print('Let us evaluate the performance of the ANN model')

print('Summary - ANN Model :')
print('=====================')
classifier.summary()
print('=====================')

# evaluating the final loss and accuracy of the classifier
test_loss, test_accuracy = classifier.evaluate(x_test, y_test)
print('Final Loss and Accuracy of the Classifier on the Testing Set')
print('============================================================')
print('loss :',round(test_loss*100),'%')
print('accuracy :',round(test_accuracy*100),'%')
print('============================================================')

# predicting the testing set results
print('Obtaining the prediction probabilities...')
y_pred = classifier.predict(x_test)
print('Categorizing the cases with probabilities above 50% as yes and below as no')
y_pred = (y_pred > 0.50)
#--------------------------------------------------------------------------------------------------------------------------------------------
# importing confusion matrix and roc_auc_score from sklearn
print('Plotting the Confusion Matrix...')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# importing seaborn for plotting the heatmap
import seaborn as sn

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
df_cm = pd.DataFrame(cm, index = ('no', 'yes'), columns = ('predicted no', 'predicted yes'))
plt.figure(figsize = (5,4))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
plt.show()
print("Test Data Accuracy: %0.4f" % roc_auc_score(y_test, y_pred))
#--------------------------------------------------------------------------------------------------------------------------------------------
# importing roc curve and metrics from sklearn
print('Plotting the ROC curve...')
from sklearn.metrics import roc_curve
import sklearn.metrics as metrics

fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
roc_auc=roc_auc_score(y_test, y_pred)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#--------------------------------------------------------------------------------------------------------------------------------------------

# Saving the ANN model
print('Saving the ANN model as predict_bank_subscription_ann.model...')
classifier.save('predict_bank_subscription_ann.model')
print('Saved ANN model Successfully')


# Syntax for loading the saved ANN model
# ann_model = keras.models.load_model('predict_bank_subscription_ann.model')
