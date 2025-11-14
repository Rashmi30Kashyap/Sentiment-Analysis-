import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



df=pd.read_csv("https://docs.google.com/spreadsheets/d/1C8FdK5HWTIcDrGSb1x-8TQkv6_4UJ9rBZlOuS_phWqY/export?format=csv&usp=sharing")
x=df['summary']
mymodel=SentimentIntensityAnalyzer()
l=[]
for k in x:
    pred=mymodel.polarity_scores(k)
    if pred['compound']>0.01:
        l.append("Positive")
    elif pred['compound']< -0.01:
        l.append(' Negative')
    else:
        l.append('Neutral')
df['Sentiment']=l

print(df)
df.to_csv("/Users/akshitkashyap/Desktop/RASHMI/Job_projects/projects/project_3/results.csv",index=False)

















































   
         
