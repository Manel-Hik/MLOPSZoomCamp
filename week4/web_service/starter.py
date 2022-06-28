#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[1]:


import pickle
import pandas as pd


# In[2]:


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


# In[4]:


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[5]:


df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-02.parquet')


# In[6]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = lr.predict(X_val)


# Q1 Solution

# In[7]:


y_pred.mean()


# Q2: Preparing the output

# In[18]:


from datetime import datetime
year= datetime.today().year
month=datetime.today().month
df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df["prediction"]=y_pred
df_result = df[["ride_id" , "prediction"]]
df_result.to_parquet(
    "df_result",
    engine='pyarrow',
    compression=None,
    index=False
)


# Q3. Creating the scoring script

# In[19]:


get_ipython().system('jupyter nbconvert starter.ipynb --to script')


# Q4 Solution:
# 

# #first hash for scikit learn 
# #"sha256:08ef968f6b72033c16c479c966bf37ccd49b06ea91b765e1cc27afefe723920b"

# Q5. Parametrize the script

# In[17]:


def get_data_with_date(month,year):
    datetime_object1 = datetime.strptime(month,'%B')
    datetime_object2 = datetime.strptime(year,'%Y')
    
    filename= "https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_"+ str(datetime_object2.year) + "-" + str(datetime_object1.month).zfill(2)+".parquet"
  
    df = read_data(filename)
    return df


# In[12]:


def get_mean_predicted_duration(df):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)
    return(y_pred.mean())


# In[14]:


get_ipython().system('python starter.py March 2021')


# In[ ]:





# 
