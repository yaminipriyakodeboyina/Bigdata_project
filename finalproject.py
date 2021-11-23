#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
# from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# In[2]:


data=pd.read_csv('data1.csv')


# In[3]:


df=data.copy()
df1=data.copy()
XX = df1.drop(["price"],axis = 1)
YY = df1["price"]

df.drop(df[df.price <= 0].index,inplace=True)


# ax = sns.histplot(df['price'], kde=True)
ax = sns.kdeplot(df['price'])
ax.set_title('Disribution of Price')
# plt.show()
plt.savefig('Disribution_of_Price_before_logtransform.png')
plt. clf() 

# df['price'] = df['price'].replace([data['price'][np.abs(stats.zscore(data['price'])) > 3]],np.median(df['price']))
df['price'] = np.log(df['price'])


# In[9]:


# ax = sns.histplot(df['price'], kde=True)
ax = sns.kdeplot(df['price'])
ax.set_title('Disribution of Price')
# plt.show()
plt.savefig('Disribution_of_Price_after_logtransform.png')
plt. clf() 

# ### Bedrooms is Continuous variable. I did changes for rows with 0 bedrooms. Explained below

# #### Bedroom data looks good and it is correlated with price data, so It is one of the main feature to predict price. But few rows has 0 bedrooms 

# #### It is not normal to have no bedrooms in a housse that costs 1095000.0 and 1295648.0	

# #### houses which has 0 bedrooms has nearly same sqft_living as 7 bedrooms, so we will replace 0 bedrooms with 4 bedrooms

df['bedrooms'].replace(to_replace = 0, value = 7, inplace = True)


# ### Bathroom is a Continuous Variable. I did some changes for rows with 0 bathrooms as follows:

# #### same as bedrooms,houses which has 0 bathrooms has nearly same sqft_living as 4 bathrooms, so we will replace 0 bathrooms with 4 bathrooms

# In[11]:


df['bathrooms'].replace(to_replace = 0, value = 4, inplace = True)


# #### bedrooms values with more than 7 are outliers, so we make bedroom value more than 7 to 7

# In[12]:


df['bedrooms'] = np.where((df.bedrooms >7 ), 7, df.bedrooms)


# #### bathrooms values with more than 6 are outliers, so we make bathrooms value more than 6 to 6

# In[13]:


df['bathrooms'] = np.where((df.bathrooms >6 ), 6, df.bathrooms)


# #### sqft_living values with more than 6000 are outliers, so we make sqft_living value more than 6000 to 6000

# In[14]:


df['sqft_living'] = np.where((df.sqft_living >6000 ), 6000, df.sqft_living)


# #### sqft_lot values with more than 300000 are outliers, so we make sqft_lot value more than 300000 to 300000

# In[15]:


df['sqft_lot'] = np.where((df.sqft_lot >300000 ), 300000, df.sqft_lot)
# display(df[df.sqft_lot > 300000])


# #### sqft_above values with more than 5000 are outliers, so we make sqft_above value more than 5000 to 5000

# In[16]:


df['sqft_above'] = np.where((df.sqft_above >5000 ), 5000, df.sqft_above)
# display(df[df.sqft_above > 5000])


# #### sqft_basement values with more than 2000 are outliers, so we make sqft_basement value more than 2000 to 2000

# In[17]:


df['sqft_basement'] = np.where((df.sqft_basement >2000 ), 2000, df.sqft_basement)
# display(df[df.sqft_basement > 2000])


# In[18]:


# df.price.max()


# In[19]:


ax = sns.pairplot(df)
plt.savefig('compare_features_aftercleaning.png')
plt.clf()
ax = sns.pairplot(df1)
plt.savefig('compare_features_beforecleaning.png')
plt.clf()


# ## Split the data to train and test set

# In[20]:


# df.drop(['street','city','country','date','sqft_above','waterfront','yr_renovated'],axis=1, inplace=True)
df.drop(['street','city','country','date'],axis=1, inplace=True)

df.to_csv("cleaned_houseprice_data.csv", sep='\t')
# In[21]:


# X1 = df.drop(["price"],axis = 1)
# Y = df["price"]


# In[22]:





# X1= pd.get_dummies(X1, columns=['statezip'], prefix = ['statezip'])


# # In[25]:


# imp=SimpleImputer()
# ct=make_column_transformer(
# (imp,['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','view','sqft_basement','yr_built','sqft_above','waterfront','yr_renovated']),
#     remainder='passthrough')


# # In[26]:


# X=pd.DataFrame(ct.fit_transform(X1))



# X.columns=X1.columns


# # ### Linear Regression without Scalling


# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, random_state=0)




# lr = LinearRegression()
# print("Cross Validation scrore for linear regression without scalling",cross_val_score(lr, X, Y))
# lr.fit(X_train, y_train)
# print("score for train data",lr.score(X_train, y_train))
# print("score for test data",lr.score(X_test, y_test))


# #Ridge without Scalling



# clf=Ridge(alpha=1.0)
# print("Cross Validation scrore for Ridge without scalling",cross_val_score(clf, X, Y))
# clf.fit(X_train,y_train)

# print("score for train data",clf.score(X_train, y_train))
# print("score for test data",clf.score(X_test, y_test))


# # LinearRegression,ElasticNet, Lasso, Ridge after scalling




# numeric_features =['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','view','sqft_basement','yr_built']
# categorical_features  = ['statezip']

# numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
# categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorical_features)])
# clf = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('classifier', LinearRegression())])
# X = df.drop(["price"],axis = 1)
# Y=df['price']
# print("Cross Validation for LinearRegression",cross_val_score(clf, X, Y, cv=5))
# x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)
# clf.fit(x_train, y_train)
# print(clf.score(x_train, y_train))
# print(clf.score(x_test, y_test))


# clf = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('classifier',Ridge(alpha=1.0))])
# print("Cross Validation for Ridge",cross_val_score(clf, X, Y, cv=5))
# clf.fit(x_train,y_train)
# print("score for train data",clf.score(x_train, y_train))
# print("score for test data",clf.score(x_test, y_test))
numeric_features =['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','view','sqft_basement','yr_built']
categorical_features  = ['statezip']

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LinearRegression())])
X = df.drop(["price"],axis = 1)
Y=df['price']
print("results after cleaning data")
print("Cross Validation for LinearRegression",cross_val_score(clf, X, Y, cv=5))
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)
clf.fit(x_train, y_train)
print("Performance score for test data",clf.score(x_test, y_test))
print("Performance score for train data",clf.score(x_train, y_train))

print("data used here is not cleaned")
print("Cross Validation for LinearRegression",cross_val_score(clf, XX, YY, cv=5))
x_train, x_test, y_train, y_test = train_test_split(XX, YY, random_state=0)
clf.fit(x_train, y_train)
print("Performance score for test data",clf.score(x_test, y_test))
print("Performance score for train data",clf.score(x_train, y_train))


