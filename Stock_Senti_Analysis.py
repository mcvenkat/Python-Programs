import numpy as np
import pandas as pd
from textBlob import TextBlob
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrxics import accuracy_score, classification_report
from sklearn.discriminantanalysiz import LinearDiscriminantAnalysis

#Load the dataset
from google.colab import files
files.upload()

#Store the data into variables
df1 = pd.read_csv('')
df2 = pd.read_csv('')

#Merge the datasets on the date field
merge = df1.merge(df2, how = 'inner', on= 'Date', left_index = True)

#Combine the top news headlines
headlines = []
for row in range(0, len(merge.index)):
    headlines.append(' '.join(str(x) for x in merge.iloc[row, 2:27]))

#Clean the data
clean_headlines = []
for i in range(0, len(headlines)):
    clean_headlines.append(re.sub("b[(')]", '', headlines[i])) #remove b'
    clean_headlines.append(re.sub('b['")]', '', headlines[i]]))
    clean_headlines.append(re.sub("\'", '', clean_headlines[i]))

#Add the clean headlines to the Merge
merge['Combined_News'] = clean_headlines

#Create a function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

#Create a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

#Create two columns Subjectivity and Polarity
merge['Subjectivity'] = merge['Combined_News'].apply(getSubjectivity)
merge['Polarity'] = merge['Combined_News'].apply(getPolarity)

#Create a function to get sentiment scores
def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment

#Get the sentiment scores for each day
compound = []
neg = []
pos = []
neu = []
SIA = 0

for i in range(0, len(merge['Combined_News'])):
    SIA = getSIA(merge['Combined_News'][i])
    compound.append(SIA['compound'])
    neg.append(SIA['neg'])
    neu.append(SIA['neu'])
    pos.append(SIA['pos'])

#Store the sentiment scores in the merge dataset
merge['Compound'] = compound
merge['Negative'] = neg
merge['Neutral'] = neu
merge['Positive'] = pos

merge.head(5)

#Create a list of columns to keep
keep_columns = ['Open', 'High', 'Low', 'Volume', 'Subjectivity', 'Polarity', 'Compound', 'Positive', 'Neutral', 'Negative', 'Label']
df = merge[keep_columns]
df

#Create the feature dataset
X = np.array(df.drop['Label'], 1)
Y = np.array(df['Label'])

#Split the data into 80% training and 20% testing
x_train, y_train, x_test, y_test = train_test_split(X, Y, test = 0.2, random_state = 0)

#Create and train the model
model = LinearDiscriminantAnalysis().fit(x_train, y_train)

#Show the predictions
predictions = model.predict(x_test)
predictions

#Show the model metrics
print(classification_report(y_test, predictions))
