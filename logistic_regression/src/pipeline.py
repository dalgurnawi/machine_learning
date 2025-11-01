# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
import datetime as dt
from datetime import datetime
import os

class Pipeline:
    def __init__(self, input_path, input_fraud_city_path, bandwidth_plus, bandwidth_minus, output_path):
        self.input_path = input_path
        self.input_fraud_city_path = input_fraud_city_path
        self.bandwidth_plus = bandwidth_plus
        self.bandwidth_minus = bandwidth_minus
        self.output_path = output_path 

    def pipeline(self,input_path, input_fraud_city_path, bandwidth_plus, bandwidth_minus, output_path):
        # Change directory to the data location
        # Read the data
        df = pd.read_csv(input_path)
        fraud_rates_cities = pd.read_csv(input_fraud_city_path)

        # Drop unnecessary columns
        fraud_rates_cities=fraud_rates_cities.drop(columns=['NotFraud', 'Fraud', 'Population'])

        # Create Fraudulent MerchantID DF
        fraud = df[df['IsFraud']==1].groupby(by='MerchantID').count().reset_index()
        fraud = fraud[['MerchantID', 'IsFraud']]
        fraud = fraud.rename(columns={'IsFraud':'FraudCount'})

        # Split Datetime
        df["Year"] =(df["TransactionDate"].astype(str).str.split(" ").str[0]).str.split('-').str[0]
        df["Month"]=(df["TransactionDate"].astype(str).str.split(" ").str[0]).str.split('-').str[1]
        df["Day"]=(df["TransactionDate"].astype(str).str.split(" ").str[0]).str.split('-').str[2]
        df['hour']=(df["TransactionDate"].astype(str).str.split(" ").str[1]).str.split(':').str[0]
        df["Year-Month"] = df["Year"]+'-'+df["Month"]

        # Create Threshold DF
        monay_mean = df[["Amount", "Year-Month"]]
        monay_mean= monay_mean.rename(columns={'Amount':'Mean'})
        monay_mean = monay_mean.groupby(by=['Year-Month']).mean().reset_index()

        df = pd.merge(df, monay_mean, on='Year-Month')

        df['Mean_plus'] = df["Mean"]*bandwidth_plus
        df['Mean_minus'] = df["Mean"]*bandwidth_minus

        df['un_amount'] = np.where(((df['Amount'] < df['Mean_minus']) |(df['Amount'] > df['Mean_plus'])),1,0)
        df['TransactionType'] = np.where(df['TransactionType']=='refund', 1, 0)

        # Standardise the amounts
        df['standardised_amount']= (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

        df = pd.merge(df,fraud_rates_cities,on='Location')
        df = pd.merge(df,fraud, on='MerchantID')
        df = df.drop(columns=['TransactionID','TransactionDate','MerchantID','Location','Year-Month','Mean','Mean_plus','Mean_minus','Amount','Year'])

        df.to_csv(output_path, index=False)