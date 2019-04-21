import pandas as pd
dt_Test = pd.read_csv('drugsComTest_raw.tsv',delimiter='\t') 
dt_Train = pd.read_csv('drugsComTrain_raw.tsv', delimiter='\t')

dt_Test.columns = ['Id','drugName','condition','review','rating','date','usefulCount']
dt_Train.columns = ['Id','drugName','condition','review','rating','date','usefulCount']

from nltk.corpus import stopwords
stopwords = stopwords.words('english')

# dt_Train = dt_Train[0:100]
# dt_Test = dt_Test[0:100]

dt_Train['cleanReview'] = dt_Train['review'].apply(lambda x: ' '.join([
                                                item for item in x.split() 
                                                if item not in stopwords])) 

dt_Test['cleanReview'] = dt_Test['review'].apply(lambda x: ' '.join([
                                                item for item in x.split() 
                                                if item not in stopwords]))

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Process file Train
dt_Train['vaderReviewScore'] = dt_Train['cleanReview'].apply(lambda x: analyzer.polarity_scores(x)['compound'])+10

dt_Train.loc[dt_Train['rating'] >=7.0,"ratingSentimentLabel"] = 2
dt_Train.loc[(dt_Train['rating'] >=4.0) & (dt_Train['rating']<7.0),"ratingSentimentLabel"]= 1
dt_Train.loc[dt_Train['rating']<=3.0,"ratingSentimentLabel"] = 0

dt_Train = dt_Train[['Id', 'vaderReviewScore', 'ratingSentimentLabel']]
# dt_Train
dt_Train.to_csv('DrugsTrainProcessed.csv')

#Process file Test
dt_Test['vaderReviewScore'] = dt_Test['cleanReview'].apply(lambda x: analyzer.polarity_scores(x)['compound'])+10

dt_Test.loc[dt_Test['rating'] >=7.0,"ratingSentimentLabel"] = 2
dt_Test.loc[(dt_Test['rating'] >=4.0) & (dt_Test['rating']<7.0),"ratingSentimentLabel"]= 1
dt_Test.loc[dt_Test['rating']<=3.0,"ratingSentimentLabel"] = 0

dt_Test = dt_Test[['Id', 'vaderReviewScore', 'ratingSentimentLabel']]
dt_Test
dt_Test.to_csv('DrugsTestProcessed.csv')
