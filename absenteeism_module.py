#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self,X,y=None):
        self.scaler.fit(X[self.columns],y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self,X,y=None,copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]),columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled,X_scaled], axis=1)[init_col_order]
    
class absenteeism_model():
    
    def __init__(self, model_file, scaler_file):
        # read the saved model and scaler files
        with open('model','rb') as model_file, open('scaler','rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
        
    # take a data file (*.csv) and preprocess it
    def load_and_clean_data(self, data_file):
        
        # import the data file
        df = pd.read_csv(data_file, delimiter=',')
        # store data
        self.df_with_predictions = df.copy()
        # to preserve the code flow, make 'Absenteeism Time in Hours' column with NaN
        df['Absenteeism Time in Hours'] = 'NaN'
        
        df['reason_1'] = df['Reason for Absence'].map(lambda x: 1 if x in range(1,15) else 0)
        df['reason_2'] = df['Reason for Absence'].map(lambda x: 1 if x in range(15,18) else 0)
        df['reason_3'] = df['Reason for Absence'].map(lambda x: 1 if x in range(18,22) else 0)
        df['reason_4'] = df['Reason for Absence'].map(lambda x: 1 if x in range(22,29) else 0)
        
        df = df.drop(['ID', 'Reason for Absence'], axis=1)
        
        column_names_reordered = ['reason_1','reason_2','reason_3','reason_4', 'Date', 'Transportation Expense',
                                  'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
                                  'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]
        
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.dayofweek
        
        df['Education'] = df['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})
        
        df = df.drop(['Date', 'Absenteeism Time in Hours', 'Day', 'Daily Work Load Average',
                      'Distance to Work'], axis=1)
        
        df = df.fillna(value=0)
        
        self.preprocessed_data = df.copy()
        self.data = self.scaler.transform(df)
    
    
    def predicted_probability(self):
        if(self.data is not None):
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred
        
    def predicted_output_category(self):
        if(self.data is not None):
            pred_output = self.reg.predict(self.data)
            return pred_output
        
    def predicted_output(self):
        if(self.data is not None):
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data

