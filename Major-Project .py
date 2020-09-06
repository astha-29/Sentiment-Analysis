import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def user_input():
  Review = st.text_input('Enter Your Review here')
  data = {'Sentiment':Review}
  features = pd.DataFrame(data,index=[0])
  return features

st.title("Machine Learning Model")
st.subheader("SENTIMENT ANALYSIS OF AMAZON FOOD REVIEW")
dframe = user_input()
st.write(dframe)

products=pd.read_csv('/content/drive/My Drive/Colab Notebooks/ML/Reviews (1).csv')
products['Sentiment']= np.where(products['Score']>3,'Positive','Negative')
products = products.drop(['ProductId','UserId','ProfileName','Id','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time','Summary'], axis=1)

x=products.iloc[:,0].values
y=products.iloc[:,1].values
text_model = Pipeline([('tfidf',TfidfVectorizer(min_df = 5, ngram_range = (1,2))),('model',LogisticRegression())])
text_model.fit(x,y)
y_pred = text_model.predict(dframe)

st.write([y_pred])