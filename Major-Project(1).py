import streamlit as st
import pandas as pd
import numpy as np
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
st.subheader("SENTIMENT ANALYSIS ON AMAZON FOOD REVIEW")
dframe = user_input()
st.write(dframe)

df=pd.read_csv('/content/drive/My Drive/Colab Notebooks/Reviews.csv')
df['Sentiment']= np.where(df['Score']>3,'Positive','Negative')
df = df.drop(['ProductId','UserId','ProfileName','Id','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time','Summary'], axis=1)

x=df.iloc[:,0].values
y=df.iloc[:,1].values
text_model = Pipeline([('tfidf',TfidfVectorizer(min_df = 5, ngram_range = (1,2))),('model',LogisticRegression())])
text_model.fit(x,y)
y_pred = text_model.predict(dframe)

st.write([y_pred])