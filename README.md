## SENTIMENT ANALYSIS ON PATIENT’S HOSPITAL REVIEW

### Data description, collection, processing and Analysis

#### COLLECTION
First dataset “Hospital Customer Satisfaction 2016-2020” found in Kaggle, consisting review of hospitals in numbers and words - 2641 surveyor
Second dataset consists only descriptive surveys, with few questions to be answered by each surveyor – 175 surveyor   
Both the dataset used contains few personal features such as name, date of birth and responses. These features are marked as sensitive data as they provide identification of an individual
Combining the two datasets allowed the model to assess how patients felt about common medical facilities
NLTK library is utilized to manage these datasets. It provides facilities such as lemmatization, stemming and tokenization etc

#### PRE-PROCESSING
- NLTK library tools like tokenization, lemmatization, stop word removal and stemming were used to pre-process the data
  
#### DATA CLEANING
- First find and remove missing values: We got rid of columns which mostly contain missing values
- The ‘gender’ feature had three classifications: ‘male’, ‘female’ and ‘others. Values like ‘M’, ‘F’, ‘Male’, ‘Female’ and ‘Other’ will be removed to attain it.
- The missing values in ‘age’ feature can be replaced with the mean of the column. Further, for age less than 18 or greater than 90 years, we can find the median for the subset [0,18] and [90,max_value] respectively and get a better estimate for the values of each age group.
- The ‘Facility Received’ feature has 0.014% ‘NaN’ values for which we are considering ‘No’ response.
- The ‘review’ feature has 0.20% ‘NaN’ values which we are replacing with ‘Don’t know’ for readability and consistency.
- The other data containing descriptive responses such as ‘yes’, ‘no’ or ‘don’t know’ are eliminated from the dataset to employ processing techniques.



  





