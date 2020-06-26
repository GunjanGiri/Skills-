#!/usr/bin/env python
# coding: utf-8

# In[9]:


import turicreate as t


# In[10]:


data = t.SFrame("Social_Network_Ads.csv")


# In[11]:


data.head()


# In[12]:


data.show()


# In[13]:


training_set ,test_set = data.random_split(.8, seed = 0)


# In[14]:


model = t.linear_regression.create(training_set,target = 'Purchased',features=['Age'])


# In[15]:


print (test_set['Purchased'].mean())


# In[16]:


print (model.evaluate(test_set))


# In[17]:


model.coefficients


# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(test_set['Age'],test_set['Purchased'],'.',
        test_set['Age'],model.predict(test_set),'-')


# In[19]:


my_features = ['Gender','EstimatedSalary']


# In[20]:


data[my_features].show()


# In[21]:


t.show(data['Gender'],data['Purchased'])


# In[22]:


t.show(data['EstimatedSalary'],data['Purchased'])


# In[23]:


my_features_model = t.linear_regression.create(training_set,target='Purchased',features=my_features)


# In[24]:


print (my_features)


# In[25]:


print (model.evaluate(test_set))
print (my_features_model.evaluate(test_set))


# In[47]:


user_id_1 = data[data['User ID'] == "15697686"]


# In[48]:


user_id_1


# In[49]:


print (user_id_1['Purchased'])


# In[ ]:




