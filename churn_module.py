import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class DataLoader:
  def __init__(self, path):
    self.path = path
    self.df = None
  def load_data(self):
    self.df = pd.read_csv(self.path)
    return self.df
  def clean_data(self):
    self.df.dropna(inplace=True)
    bin = [0, 20, 35, 50, 65]
    label = ['<20', '21-35', '36-50', '>50']
    self.df['age_group'] = pd.cut(self.df['age'], bins=bin, labels=label, right=False)
    return self.df

class AnalyzerData:
  def __init__(self, df):
    self.df = df
  def churn_rate(self):
    return self.df['churn'].mean()*100
  def churn_rate_gender(self):
    return self.df.groupby('gender')['churn'].mean()*100
  def churn_rate_age(self):
    return self.df.groupby('age_group')['churn'].mean()*100
  def churn_rate_marital(self):
    return self.df.groupby('has_marital')['churn'].mean()*100
  def churn_rate_children(self):
    return self.df.groupby('has_children')['churn'].mean()*100
  def churn_rate_performance(self):
    return self.df.groupby('performance_score')['churn'].mean()*100
  def correlation(self):
    self.df.corr(numeric_only=True)

class ChurnModel:
  def __init__(self, df):
    self.df = df
    self.model = LogisticRegression()
  def prepare_data(self):
    x = self.df[['age', 'tenure_months', 'monthly_income', 'has_marital', 'has_children', 'performance_score']]
    y = self.df['churn']
    return train_test_split(x, y, test_size=0.2, random_state=42)
  def train(self, x_train, y_train):
    self.model.fit(x_train, y_train)
    return self.df
  def evaluate(self, x_test, y_test):
    y_pred = self.model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return acc
  def predict(self, input_data):
    df = pd.DataFrame([input_data])
    return self.model.predict(df)[0]
