#!/usr/bin/env python
# coding: utf-8

# In[60]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[61]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier


# In[62]:


from sklearn.model_selection import train_test_split


# In[87]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[64]:


data = pd.read_csv('phishing.csv')
data.head()


# In[65]:


data.shape


# In[66]:


data.columns


# In[67]:


data.info()


# In[68]:


data.hist(bins = 50,figsize = (15,15))
plt.show()


# In[69]:


plt.figure(figsize=(15,13))
sns.heatmap(data.corr())
plt.show()


# In[70]:


data.describe()


# In[71]:


#Dropping the index column
data = data.drop(['Index'], axis = 1)


# In[72]:


data.info()


# In[73]:


data.isnull().sum()


# In[74]:


data = data.sample(frac=1).reset_index(drop=True)
data.head()


# In[75]:


Y = data['class']
X = data.drop('class',axis=1)
X.shape,Y.shape


# In[76]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 2)
X_train.shape, X_test.shape


# In[77]:


Y_train.shape, Y_test.shape


# In[118]:


ML_Model = []
acc_train = []
acc_test = []


def storeResults(model, a,b):
  ML_Model.append(model)
  acc_train.append(round(a, 3))
  acc_test.append(round(b, 3))


# In[79]:


tree = DecisionTreeClassifier(max_depth = 7)

tree.fit(X_train, Y_train)


# In[80]:


Y_test_tree = tree.predict(X_test)
Y_train_tree = tree.predict(X_train)


# In[114]:


acc_train_tree = accuracy_score(Y_train,Y_train_tree)
acc_test_tree = accuracy_score(Y_test,Y_test_tree)

print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))


# In[119]:


storeResults('Decision Tree', acc_train_tree, acc_test_tree)


# In[90]:


svm = SVC(kernel='linear', C=1.0, random_state=2)

svm.fit(X_train, Y_train)


# In[91]:


Y_test_svm = svm.predict(X_test)
Y_train_svm = svm.predict(X_train)


# In[115]:


acc_train_svm = accuracy_score(Y_train,Y_train_svm)
acc_test_svm = accuracy_score(Y_test,Y_test_svm)

print("SVM: Accuracy on training Data: {:.3f}".format(acc_train_svm))
print("SVM : Accuracy on test Data: {:.3f}".format(acc_test_svm))


# In[120]:


storeResults('SVC', acc_train_svm, acc_test_svm)


# In[94]:


logisticRegr = LogisticRegression()

logisticRegr.fit(X_train, Y_train.ravel())


# In[95]:


Y_train_LR = logisticRegr.predict(X_train)
Y_test_LR = logisticRegr.predict(X_test)


# In[117]:


acc_train_lr = accuracy_score(Y_train,Y_train_LR)
acc_test_lr  =  accuracy_score(Y_test,Y_test_LR)
print("Train Accuracy: {:.3f}".format(acc_train_lr))
print("Test Accuracy: {:.3f}".format(acc_test_lr))


# In[121]:


storeResults('Logistic Reg', acc_train_lr, acc_test_lr)


# In[122]:


results = pd.DataFrame({ 'ML Model': ML_Model,    
    'Train Accuracy': acc_train,
    'Test Accuracy': acc_test})
results


# In[39]:


classifiers=[['Decision Tree :' ,DecisionTreeClassifier(max_depth = 7)],['SVC :',SVC(gamma ='auto',kernel='linear', C=1.0, random_state=12, probability=True)],['LogisticRegression :',LogisticRegression(solver ='lbfgs',max_iter = 1000)]]


# In[45]:


pred_df = pd.DataFrame()
pred_df['Actual'] = Y_test
pred_df['DT'] = Y_test_tree
pred_df['SVC'] = Y_test_svm
pred_df['LR'] = Y_test_LR


# In[46]:


pred_df


# In[47]:


from sklearn.ensemble import VotingClassifier
clf1=DecisionTreeClassifier(max_depth = 7)
clf2=SVC(gamma ='auto',kernel='linear', C=0.8, random_state=12, probability=True)
clf3=LogisticRegression(solver ='lbfgs',max_iter = 200)


# In[48]:


vot_hard = VotingClassifier(estimators = classifiers,voting = 'hard')
vot_hard.fit(X_train,Y_train)
Y_pred1 = vot_hard.predict(X_test)

Acc1 = accuracy_score(Y_test,Y_pred1)
Acc1


# In[49]:


vot_soft = VotingClassifier(estimators = classifiers,voting = 'soft')
vot_soft.fit(X_train,Y_train)
Y_pred2 = vot_soft.predict(X_test)
Acc2 = accuracy_score(Y_test,Y_pred2)
Acc2


# In[50]:


from sklearn.metrics import classification_report


# In[57]:


report1 = classification_report(Y_test, Y_pred1, output_dict=True)
df1 = pd.DataFrame(report1)
df1


# In[58]:


report2 = classification_report(Y_test, Y_pred2, output_dict=True)
df2 = pd.DataFrame(report2)
df2

