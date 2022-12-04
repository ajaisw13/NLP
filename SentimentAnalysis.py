# Sentiment Analysis 

import pandas
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nltk.sentiment import SentimentIntensityAnalyzer

stopWordList = stopwords.words('english')

def tokenization(dataframe,column_name):
    return dataframe[column_name].apply(word_tokenize)

def predictOverallSentiment(scores):
    final_sentiment=''
    if (scores['neu'] > scores['neg']) and (scores['neu'] > scores['pos']):
        final_sentiment = 'neutral'
    elif scores['pos'] > scores['neg']:
        final_sentiment ='positive'
    elif scores['neg'] > scores['pos']:
        final_sentiment = 'negetive'
    return final_sentiment

def hospitalPositveNegetiveNeutralIndex(dataframe,col_name,res_col):
    hospitals = dataframe[col_name].unique().tolist()
    
    names=[]
    positive_ind=[]
    neutral_ind=[]
    negative_ind=[]
    
    for h in hospitals:
        names.append(h.replace('\t',''))
        cnt=0
        positive=0
        neutral=0
        negative=0
        for index,row in dataframe.iterrows():
            if h == row[col_name] :
                if row[res_col] =='positive':
                    positive = positive + 1
                elif row[res_col] =='neutral':
                    neutral= neutral + 1
                else:
                    negative=negative + 1           
                cnt= cnt + 1
        positive_ind.append(positive/cnt)
        neutral_ind.append(neutral/cnt)
        negative_ind.append(negative/cnt)
    return pandas.DataFrame((list(zip(names, positive_ind,neutral_ind,negative_ind))),
               columns = ['Healthcare Facility', 'Pos_index','Neu_index','Neg_index'])
                  

# covert text to lowercase
def covertToLower(df,col):
    return list(map(lambda x: x.lower(), df[col]))


def filterOutStopWords(dataframe,column_name):
    return dataframe[column_name].apply(lambda item: [words for words in item if words not in stopWordList])


def generatePositiveNeutralNegative(dataframe,column_name):
    
    positive=[]
    neutral=[]
    negative=[]
    overall_scores=[]
    sentimentAnalyzer = SentimentIntensityAnalyzer()
    for i in dataframe[column_name]:
        scores = sentimentAnalyzer.polarity_scores(' '.join(i))
        positive.append(scores['pos'])
        neutral.append(scores['neu'])
        negative.append(scores['neg'])
        overall_scores.append(sentimentAnalyzer.polarity_scores(' '.join(i)))
        
    dataframe['sentiment_score'] =  overall_scores
    dataframe['pos'] =  positive
    dataframe['neu'] =  neutral
    dataframe['neg'] =  negative
    return dataframe

print('Hospital Facilities Feedback Data')
df_hospital_reviews = pandas.read_excel('Hospitals.xlsx', 'TestData')
df_hospital_reviews['What do you say for satisfaction you with the treatment you got in your each visit?'] = covertToLower(df_hospital_reviews,'What do you say for satisfaction you with the treatment you got in your each visit?')
df_hospital_reviews['How was the hygiene of the place you visited'] = covertToLower(df_hospital_reviews, 'How was the hygiene of the place you visited')
df_hospital_reviews['What was staffs behaviour and attitude towards'] = covertToLower(df_hospital_reviews, 'What was staffs behaviour and attitude towards')
df_hospital_reviews['What do you feel about the labrotories facilities and their equipments use?'] = covertToLower(df_hospital_reviews, 'What do you feel about the labrotories facilities and their equipments use?') 
df_hospital_reviews['Was the care provided to the patient'] = covertToLower(df_hospital_reviews, 'Was the care provided to the patient')


# Applying tokenization in all hospital feedbacks
df_hospital_reviews['text_after_tokenization'] = tokenization(df_hospital_reviews,'What do you say for satisfaction you with the treatment you got in your each visit?') \
+ tokenization(df_hospital_reviews, 'How was the hygiene of the place you visited') + tokenization(df_hospital_reviews, 'What was staffs behaviour and attitude towards')\
+ tokenization(df_hospital_reviews, 'What do you feel about the labrotories facilities and their equipments use?') \
+ tokenization(df_hospital_reviews, 'Was the care provided to the patient') 

# filtering out all the stop words from the data
df_hospital_reviews['text_after_tokenization'] = filterOutStopWords(df_hospital_reviews,'text_after_tokenization')


#Using Sentiment Analysis model from NLTK
df_hospital_reviews=generatePositiveNeutralNegative(df_hospital_reviews,'text_after_tokenization')
df_hospital_reviews['res']=df_hospital_reviews['sentiment_score'].apply(predictOverallSentiment)
final_df_hospital_reviews = hospitalPositveNegetiveNeutralIndex(df_hospital_reviews,'Healthcare Facility you recently visited','res')
print(final_df_hospital_reviews.head(10))


