#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv(r"C:\Users\hp\Desktop\heart.csv")


# In[3]:


data.isnull().sum()


# In[4]:


data_dup=data.duplicated().any()


# In[5]:


data_dup


# In[6]:


data_dup=data.duplicated().any()


# In[7]:


data=data.drop_duplicates()


# In[8]:


data_dup


# In[9]:


cate_val=[]
cont_val=[]
for column in data.columns:
    if data[column].nunique() <=10:
        cate_val.append(column)
    else:
        cont_val.append(column)


# In[10]:


cate_val


# In[11]:


cont_val


# In[12]:


cate_val


# In[13]:


data['cp'].unique()


# In[14]:


cate_val.remove('sex')
cate_val.remove('target')
data=pd.get_dummies(data,columns=cate_val,drop_first=True)


# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


st=StandardScaler()
data[cont_val]=st.fit_transform(data[cont_val])


# In[17]:


data.head()


# In[18]:


X=data.drop('target',axis=1)


# In[19]:


y=data['target']


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[22]:


from sklearn.linear_model import LogisticRegression


# In[23]:


log=LogisticRegression()
log.fit(X_train,y_train)


# In[24]:


y_pred1=log.predict(X_test)


# In[25]:


from sklearn.metrics import accuracy_score


# In[26]:


accuracy_score(y_test,y_pred1)


# In[27]:


from sklearn import svm


# In[28]:


svm=svm.SVC()


# In[29]:


svm.fit(X_train,y_train)


# In[30]:


y_pred2=svm.predict(X_test)


# In[31]:


accuracy_score(y_test,y_pred2)


# In[32]:


from sklearn.neighbors import KNeighborsClassifier


# In[33]:


knn=KNeighborsClassifier()


# In[34]:


knn.fit(X_train,y_train)


# In[35]:


y_pred3=knn.predict(X_test)


# In[36]:


accuracy_score(y_test,y_pred3)


# In[37]:


score = []
for k in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    score.append(accuracy_score(y_test,y_pred))


# In[38]:


import matplotlib.pyplot as plt


# In[39]:


plt.plot(score)
plt.xlabel("K Value")
plt.ylabel("Ace")
plt.show()


# In[40]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
accuracy_score(y_test,y_pred)


# In[41]:


data=pd.read_csv(r"C:\Users\hp\Desktop\heart.csv")


# In[42]:


data=data.drop_duplicates()


# In[43]:


X=data.drop('target',axis=1)
y=data['target']


# In[44]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[45]:


from sklearn.tree import DecisionTreeClassifier


# In[46]:


dt=DecisionTreeClassifier()


# In[47]:


dt.fit(X_train,y_train)


# In[48]:


y_pred4=dt.predict(X_test)


# In[49]:


accuracy_score(y_test,y_pred4)


# In[50]:


from sklearn.ensemble import RandomForestClassifier


# In[51]:


rf=RandomForestClassifier()


# In[52]:


rf.fit(X_train,y_train)


# In[53]:


y_pred5=rf.predict(X_test)


# In[54]:


accuracy_score(y_test,y_pred5)


# In[55]:


from sklearn.ensemble import GradientBoostingClassifier


# In[56]:


gbc=GradientBoostingClassifier()


# In[57]:


gbc.fit(X_train,y_train)


# In[58]:


y_pred6=gbc.predict(X_test)


# In[59]:


accuracy_score(y_test,y_pred6)


# In[60]:


final_data=pd.DataFrame({'Models':['LR','SVM','KNN','DT','RF','GB'],
                        'ACC':[accuracy_score(y_test,y_pred1)*100,
                              accuracy_score(y_test,y_pred2)*100,
                              accuracy_score(y_test,y_pred3)*100,
                              accuracy_score(y_test,y_pred4)*100,
                              accuracy_score(y_test,y_pred5)*100,
                              accuracy_score(y_test,y_pred6)*100]})


# In[61]:


final_data


# In[62]:


import seaborn as sns


# In[63]:


sns.barplot(final_data['Models'],final_data['ACC'])


# In[64]:


X=data.drop('target',axis=1)
y=data['target']


# In[65]:


from sklearn.ensemble import RandomForestClassifier


# In[66]:


rf=RandomForestClassifier()
rf.fit(X,y)


# In[67]:


import pandas as pd


# In[68]:


new_data=pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trtbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalachh':168,
    'exng':0,
    'oldpeak':1.0,
    'slp':2,
    'caa':2,
    'thall':3,
},index=[0]) 


# In[69]:


new_data


# In[70]:


p=rf.predict(new_data)
if p[0]==0:
    print("No Disease")
else:
    print("Disease")


# In[71]:


import joblib


# In[72]:


joblib.dump(rf,'model_joblib_heart')


# In[73]:


model=joblib.load('model_joblib_heart')


# In[80]:


model.predict(new_data)


# In[ ]:





# In[ ]:




