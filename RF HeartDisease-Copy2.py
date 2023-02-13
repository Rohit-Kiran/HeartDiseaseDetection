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


# In[74]:


model.predict(new_data)


# In[75]:


from tkinter import *
import joblib
def show_entry_fields():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=int(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=int(e8.get())
    p9=int(e9.get())
    p10=float(e10.get())
    p11=int(e11.get())
    p12=int(e12.get())
    p13=int(e13.get())
    model = joblib.load('model_joblib_heart')
    result=model.predict([[p1,p2,p3,p4,p5,p6,p7,p8,p8,p10,p11,p12,p13]])
    
    if result == 0:
        Label(master, text="No Possibility of Heart Disease",bg = "white", fg = "dark green").grid(row=31)
    else:
        Label(master, text=" Possibility of Heart Disease ",bg = "white", fg = "red").grid(row=31)
    
    
master = Tk()
master.title("Heart Disease Prediction System")


label = Label(master, text = "Heart Disease Prediction System"
                          , bg = "black", fg = "white",width=100). \
                               grid(row=0,columnspan=2)


Label(master, text="Enter Your Age").grid(row=1)
Label(master, text="Male Or Female [1/0]").grid(row=2)
Label(master, text="ChestPain (0-3)").grid(row=3)
Label(master, text="Rest BPS").grid(row=4)
Label(master, text="Cholestrol").grid(row=5)
Label(master, text="Fasting Blood Sugar (in mg/l)").grid(row=6)
Label(master, text="RestECG").grid(row=7)
Label(master, text="Maximum Heart Rate").grid(row=8)
Label(master, text="Exercise Induced Angina").grid(row=9)
Label(master, text="OldPeek").grid(row=10)
Label(master, text="Hear Rate Slope").grid(row=11)
Label(master, text="Major Vessels (0-3)").grid(row=12)
Label(master, text="Thalassemia").grid(row=13)



e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)

e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
e3.grid(row=3, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
e7.grid(row=7, column=1)
e8.grid(row=8, column=1)
e9.grid(row=9, column=1)
e10.grid(row=10, column=1)
e11.grid(row=11, column=1)
e12.grid(row=12, column=1)
e13.grid(row=13, column=1)



Button(master, text='Predict',bg = "cyan", fg = "black", command=show_entry_fields).grid()

mainloop()


# In[ ]:





# In[ ]:




