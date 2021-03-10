# **Predicting Stocks with Financial News with LSTM, Differential Privacy and Sentiment Analysis**


---

Team member: Shamal Lalvani and Yunan Wu

# **Tutorial Overview**

This tutorial is divided into four parts; they are:

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#1-introduction">Introduction</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


* [1. Introduction](#**1-introduction**)
    * 1.1 What is Sentiment Analysis?
    * 1.2	What is Differential Privacy?
    * 1.3 What is ARMA?
    * 1.4 What is LSTM?
* [2. The Structure of the Model](#**2.-the-structure-of-the-model**)
* [3. Implementing the Model ](#**3.-implementung-the-model**)
     * 3.1 Dataset
     * 3.2 News Dataset Preprocess
     * 3.3 Feature Engineering
     * 3.4 Simplifying Sentiment Analysis using VADER in Python (on Social Media Text)
     * 3.5 Boeing Stock Prediciton
     * 3.6 Add Noise to the Dat
     * 3.7 Train LSTM model on stock prediciton from a single company (BA)
     * 3.8 Evaluation on the Testing Dataset
     * 3.9 Train LSTM model on stock prediciton from all companies
     * 3.10 Train LSTM model on stock prediciton without sentiment score
* [4. Summary](#**4.-summary**)


<!-- ABOUT THE PROJECT -->
# **1 Introduction**

In This post, Yunan and Shamal cover the techniques used in the paper DP-LSTM: Differential Privacy-inspired LSTM for Stock Prediction Using Financial News [#l], which aims to predict financial data with the use of financial news and historical data. The main techniques used in this paper are sentiment analysis, the most basic time-series model known as ARMA, and a neural-network known as LSTM. In addition, this paper uses techniques inspired by differential privacy to learn robustness of financial trends based off of financial news. In this post, Yunan and I summarize the techniques of this paper in a manner in which will be readily available to a novice reader. The hope is that after reading this post, you will feel comfortable understanding and implementing techniques such as Neural Networks and Sentiment Analysis to conduct their own desired analyses on financial data, or any data that evolves over time. We would like to thank the authors of the paper for making their code available on Github. 

Following the introduction, we will cover the structure of the model, implementing the model, and a summary. Before we get started, lets cover some basic topics below:


**1.1 What is Sentiment Analysis?**

  When we think of social media data, for example Facebook or Twitter, we know that many individuals share a lot of data through social media posts that include text, videos and pictures. But usually, when we look into this data, the main type of information this data conveys is not the type of information you find in a textbook. It’s heavily loaded with emotion. The types of information you find are how an individual feels [happy, sad, etc.], their outlook on things/current events/politicians [positive or negative], etc.

Sentiment Analysis is a means of classifying primarily textual data [and sometimes, even pictures and videos, such as in the field of computer vision], in order to classify the sentiment/emotion of the data. To do this, the data is usually pre-processed by removing stop-words [words that are deemed to contain very little information about sentiment, such as “is” or “the”], and can even be done through an information theoretic approach [2#]. The data is usually presented as a histogram of words [see bag of words, unigrams, bigrams, trigrams, if you would like to learn more]. Following, a classifier is trained on data and words that are already classified to positive or negative, in order to predict the sentiment of posts as being primarily positive or negative. The application of techniques in sentiment analysis, however, are not necessary just related to sentiment/emotion. For example, Yunan and I work on a project in which our team-mate uses sentiment analysis to classify if radiology medical reports of Head CT Scans allude to brain emergencies.

In the case of this paper, we would like to look at financial news and understand the sentiment toward the stock market and stocks. As the massive price shifts in cryptocurrencies and tech stocks show us over and over, it is well observed that the emotion towards financial news plays a huge role in influencing stock price over time.

**1.2	What is Differential Privacy?**

Differential Privacy is a technique used in data collection that allows companies such as Apple to use your information to understand large scale trends of data, while at the same time protecting your individual information from being discovered. It involves the injection of noise to your data to protect your individual information, while at the same time allowing overall trends of the data to become apparent. The main reason to use it is to understand trends of the data, while at the same time protecting individual’s privacy from which the data is obtained.  [3#]

To summarize it, “nothing about an individual should be learnable from the database that cannot be learned without access to the database.” [4#]

**1.3 What is ARMA?**

Initially described in Peter Whittle’s 1951 PhD Thesis, ARMA [or Autoregressive-Modeling-Average Model] is a time series model that describes the evolution of a system as the sum of two polynomials: a polynomial for autoregression [AR for short], and a polynomial for moving average [MA for short], hence the name ARMA.

The autoregression polynomial essentially says that at any arbitrary time t, the state of the system is a linear combination of the previous p states. The moving average polynomial is described as a linear combination of the previous q errors, and the current error to the system.

Contrary to a Markov Chain, which only depends on the state of the present to predict the future, ARMA provides a means to understand the evolution of the state of a system as a function of its previous states. [5#]

**1.4 What is LSTM?**

For a more exhaustive introduction to the mechanics of an LSTM, we recommend reading [6#], from which we present a quick summary. The idea behind LSTM is this. If you think of a movie, we usually predict what might happen in one scene based on the previous scenes. While traditional Neural Networks are not great with this, Recurrent Neural Networks are Neural Networks that are designed to do so. LSTMs are a specific type of Recurrent Neural Network. 

If you look at the sequential diagram below, you will notice that the same Neural Network [A], is used at each time point, from which temporal data [X] is input into the network. The term h [the output of the network] is used in what is referred to the cell state C, which is computed at each time point and fed into the network at the next time point, and also used for prediction at the next time point. The cell state gives us the state of the system, and h tells us how to update it. This term allows us to keep track of previous information, and present it to the network in the next iteration.

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110244736-c88af700-7f25-11eb-9f3c-326150ab59e7.png" />
</p>

The diagram above shows different gates in each iteration of the Neural Network A. We will show what each of these do below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110244794-1142b000-7f26-11eb-9aec-55907b2c7af1.png" />
</p>

As shown above, the first gate takes the previous message h, and the current data, and uses the sigmoid activation function to decide what information to forget with regard to the old cell state. Below, the next step involves deciding what new information to input into the cell state.

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110244794-1142b000-7f26-11eb-9aec-55907b2c7af1.png" />
</p>

Next, we then aggregate this information into the cell state, as shown below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110244937-c70dfe80-7f26-11eb-8e4d-40c671127708.png" />
</p>

Finally, the network gives its output, as shown below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110247162-73a0ae00-7f30-11eb-8515-f6e932ae2276.png" />
</p>

The main takeaway is that LSTM is a recurrent neural network, which allows us to take into account temporal dynamics. This ends the introduction, and we will now explain the pipeline of the model in the paper.


# **2. The Structure of the Model**

Now that we have everything detailed, we will give a quick description of the structure of the model. The paper uses VADER, a sentiment analysis tool that allows us to extract polarity of the sentiment [and it’s intensity] from textual data.

The model starts off with two ARMA models. One ARMA model is for stock price [or any desire financial parameter] at a collection of successive time points [the training data]. The test data are the remaining time points from the dataset. Another ARMA model is for the sentiment of the stock market / stock parameters [from financial news], and that similarly is partitioned into train and test data. Our overall model is thus a linear combination of these two ARMA models, plus a constant.

We now have an optimization problem. We want to find the parameters of the linear combination that minimize the sum of the squares of the errors at each time point [between our ARMA model and the training data]. This is where LSTM comes in. Incorporating this into the loss function, we use the LSTM to train our temporal data and optimize the network from which we can do predictions. The model is shown below.

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110247305-37ba1880-7f31-11eb-836e-9e1d09365fa4.png" />
</p>


Note that in the model above, the Joint ARMA Model data goes into the loss function that is used in training the LSTM. Now the missing link, which we haven’t shown above, is the Differential Privacy component. Data on each dimension is injected with noise, which is drawn from a normal distribution, each with different mean and variance.

# **3. Implementing the Model**

Here, we put the codes and apply it with our dataset:

**3.1 Dataset**

- Historical Stock Prices: This data contains minute-level historical prices in the past 5 years for 86 companies. 
- News articles: This data contains 29630 news articles, each of which corresponds to a specific company (e.g., Apple Inc.). The news articles are categorized by company name. Each article includes four fields: “title”, “full text”, “URL” and “publish time”. There are 81 companies in total, all the companies are included in data 1.

**3.2 News Dataset Preprocess**

- Load all .json files use json.loads(), basically each .json file only contains 1 line

```
json_list = []

dirpath = 'path for the /stock/'
for filePath in list_files(dirpath):
     if filePath.endswith('.json'):
            with open(filePath) as f:
                for line in f:
                    data = json.loads(line)
                    for j in range(len([*data])):
                        for k in range(len(data[[*data][j]])):
                            json_list.append([data[[*data][j]][k]['pub_time'], [*data][j], data[[*data][j]][k]['title'], data[[*data][j]][k]['text'], data[[*data][j]][k]['url']])

```

Make sure the length of json list matches the total files.

```
len(json_list) # 29630
```

Convert Json to DataFrame in order to perform data analysis.

```
col_names =  ['published_date','company','title','body','url']
df= pd.DataFrame(json_list,columns=col_names)
```

Sort the data by date and save the file.
```
df = df.sort_values(by=['published_date'], ascending=True)
df=df.reset_index(inplace=False)
df.to_csv('new_data_articles.csv')
```

**3.3 Feature Engineering**

Find the missing data, we can fill it with anything

```
df[df.isnull().any(axis=1)]
df_missing_percentage=df.isnull().sum()/df.shape[0] *100
df=df.fillna('missing')
```

**3.4 Simplifying Sentiment Analysis using VADER in Python (on Social Media Text)**

Sentiment Analysis, or Opinion Mining, is a sub-field of Natural Language Processing (NLP) that tries to identify and extract opinions within a given text. The aim of sentiment analysis is to gauge the attitude, sentiments, evaluations, attitudes and emotions of a speaker/writer based on the computational treatment of subjectivity in a text.

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media. VADER uses a combination of A sentiment lexicon is a list of lexical features (e.g., words) which are generally labelled according to their semantic orientation as either positive or negative. For more details, please refer to: [sentiment analysis](https://medium.com/analytics-vidhya/simplifying-social-media-sentiment-analysis-using-vader-in-python-f9e6ec6fc52f)

```
import nltk
nltk.downloader.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

get_ipython().run_cell_magic('time', '', 'title_score = [sid.polarity_scores(sent) for sent in df.title]')

compound=[]
neg=[]
neu=[]
pos=[]

for i in range(len(title_score)):
    compound.append(title_score[i]['compound'])
    neg.append(title_score[i]['neg'])
    neu.append(title_score[i]['neu'])
    pos.append(title_score[i]['pos'])
    
df['compound'] = compound
df['neg'] = neg
df['neu'] = neu
df['pos'] = pos
```
After this, you will generate the sentiment score for each company, like this:

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110325511-0d259980-7fdd-11eb-91fe-651ee10ebb3a.png" />
</p>

**3.5 Boeing Stock Prediciton**

Specifically, we select Boeing (BA) and predict its stock price.  
Here, we import the stock price for BA.

```
sp = pd.read_csv("path/historical_price/BA_2015-12-30_2021-02-21_minute.csv")#,index_col=0)
sp=pd.DataFrame(sp)
```
And the sentiment analysis for BA and group it by days.

```
dff1=df[(df.company=='BA')]
dff1g=dff1.groupby(['published_date']).agg(['mean'])
```
Next, we need to fuse these two information together by days. As the stock price is based on minutes, we choose the last day point as the close price for each day.
```
for i in range(0,d1.shape[0]):
    t=d1['published_date'][i]
    timeStruct = time.strptime(t, "%m/%d/%Y") 
    d1['published_date'][i] = time.strftime("%Y-%m-%d", timeStruct) 

sp1=sp.copy()
sp1.drop_duplicates(subset=['Date'], keep='last', inplace = True)
```
Then, Union these two tables together:
```
date_union_1=pd.DataFrame(columns=('idx','date','mean_compound','comp_flag'))
sp_len=sp1.shape[0]
d_len=d1.shape[0]
d=d1.copy()
for i in range(0,sp_len):
    idx=i
    date=sp1['Date'][i]
    j=0
    t=0
    while j<d_len:
        if sp1['Date'][i]==d['published_date'][j]:
            mean_compound=d['compound'][j]
            comp_flag=1
            t=1
            break
        j=j+1
    if t==0:
        mean_compound=0
        comp_flag=0
    
    date_union_1=date_union_1.append(pd.DataFrame({'idx':[idx],
                                                  'date':[date],
                                                  'mean_compound':[mean_compound],
                                                  'comp_flag':[comp_flag]
        
    }),ignore_index=True)
```

**3.6 Add Noise to the Data**

Next, we add some noise to the sentiment score, to make the model more robust。 The noise variance is calculated and added to the score.

```
wsj_var=np.var(df.mean_compound)
mu=0
noise=0.1
sigma_wsj=noise*wsj_var
df_noise['noise']=df['mean_compound']

for i in range(0,n):
    df_noise['noise'][i]+=np.random.normal(mu,sigma_wsj)
    
```

**3.7 Train LSTM model on stock prediciton from a single company (BA)**

After we prepare the sentiment score with noise and the stock price. We can start building the model!

We split the dataset into 85% for training and the rest 15% for validation. And below are the settings of parameters:

```
split = (0.85)
sequence_length=10;
normalise= True
batch_size=100;
input_dim=2
input_timesteps=9
neurons=50
epochs=5
prediction_len=1
dense_output=1
drop_out=0
```

And choose the time window:

```
for win_i in range(0,win_num):
    normalised_window = []
    for col_i in range(0,1):#col_num):
        temp_col=window_data[win_i,:,col_i]
        temp_min=min(temp_col)
        if col_i==0:
            record_min.append(temp_min)#record min
        temp_col=temp_col-temp_min
        temp_max=max(temp_col)
        if col_i==0:
            record_max.append(temp_max)#record max
        temp_col=temp_col/temp_max
        normalised_window.append(temp_col)
    for col_i in range(1,col_num):
        temp_col=window_data[win_i,:,col_i]
        normalised_window.append(temp_col)
    normalised_window = np.array(normalised_window).T
    normalised_data.append(normalised_window)
normalised_data=np.array(normalised_data)

data_windows=normalised_data
x_train1 = data_windows[:, :-1]
y_train1 = data_windows[:, -1,[0]]
print('x_train1.shape',x_train1.shape)
print('y_train1.shape',y_train1.shape)
```
You can find that the size of the training dataset is (1149, 9, 2), where 1149 is the number of days, 9 denotes the first 9 days and 2 are the features. The size of labels is (1149,1). Similarly for the testing dataset, the size is (195, 9, 2).

Finally, we build the model here, which contains three LSTM layers and one dense layer. The loss function is mean square error and the optimizer is adam:
```
model = Sequential()
model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences = True))
model.add(Dropout(drop_out))
model.add(LSTM(neurons,return_sequences = True))
model.add(LSTM(neurons,return_sequences =False))
model.add(Dropout(drop_out))
model.add(Dense(dense_output, activation='linear'))
# Compile model
model.compile(loss='mean_squared_error',
                optimizer='adam')
# Fit the model
model.fit(x_train,y_train,epochs=20,batch_size=batch_size)
```

**3.8 Evaluation on the Testing Dataset**

```
data=x_test
prediction_seqs = []
window_size=sequence_length
pre_win_num=int(len(data)/prediction_len)

for i in range(0,pre_win_num):
    curr_frame = data[i*prediction_len]
    predicted = []
    for j in range(0,prediction_len):
        temp=model.predict(curr_frame[newaxis,:,:])[0]
        predicted.append(temp)
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
    prediction_seqs.append(predicted)

de_predicted=[]
len_pre_win=int(len(data)/prediction_len)
len_pre=prediction_len

m=0
for i in range(0,len_pre_win):
    for j in range(0,len_pre):
        de_predicted.append(prediction_seqs[i][j][0]*record_max[m]+record_min[m])
        m=m+1
print(de_predicted)
```

Accuracy, MSE and loss on testing dataset:

```
error = []
diff=y_test.shape[0]-prediction_len*pre_win_num

for i in range(y_test_ori.shape[0]-diff):
    error.append(y_test_ori[i,] - de_predicted[i])
    
squaredError = []
absError = []
for val in error:
    squaredError.append(val * val) 
    absError.append(abs(val))
    
error_percent=[]
for i in range(len(error)):
    val=absError[i]/y_test_ori[i,]
    val=abs(val)
    error_percent.append(val)
    
mean_error_percent=sum(error_percent) / len(error_percent)
accuracy=1-mean_error_percent
MSE=sum(squaredError) / len(squaredError)
```

For the single company prediction, we finally get MSE: 79.94, Accuracy: 0.9712 and mean error percent: 0.028.
The results demonstrates the good stock predicitons if we use the news from that company.

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110559023-16138980-8109-11eb-9f3f-bc62bd35cb24.png" />
</p>

We can plot the predictions and compare it with the true stock price.

```
import matplotlib.pyplot as plt
plt.plot(de_predicted, label = 'predicted')
plt.plot(y_test_ori, label = 'real')
plt.legend()
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110556640-8e2b8080-8104-11eb-81aa-a8082f08ca34.png" />
</p>


**3.9 Train LSTM model on stock prediciton from all companies**

Furthermore, we want to see if the stock predicitons of BA would be increased if we use the news from all other companies.

All similar steps are processed in step 3.5~3.8. And the final results on the testing dataset are:

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110558867-d9e02900-8108-11eb-9ec6-a7509b00b898.png" />
</p>

We also plot the predictions below:

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110558961-fe3c0580-8108-11eb-84df-7764cfc39abb.png" />
</p>

The comparisons demonstrate that the performance of the stock predicitons decreases if we add in sentiment scores from other companies.

**3.9 Train LSTM model on stock prediciton without sentiment score**
Finally, we trained the model only with the stock price from previous days without the sentiment score.

Here are the results:
<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110560120-0a28c700-810b-11eb-8f18-9bc693d7dab4.png" />
</p>
And we plot the predicitons below:
<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110560162-22004b00-810b-11eb-946e-d29a1f02c238.png" />
</p>

The results demonstrate that adding additional sentiment scores from that specific company did increase the model performance. 

To have a better understanding of these different methods, we plot all the predictions together:

<p align="center">
  <img src="https://user-images.githubusercontent.com/45545175/110560900-6213fd80-810c-11eb-9b77-9108b3247c85.png" />
</p>


# **4. Summary**

To summarize, we covered how sentiment and financial news, which vary with time, can be used to predict future prices of stocks, trained with a recurrent neural network, which takes into account the time dependency, known as LSTM. In addition, differential privacy is used, which adds noise to our data, which in principle can be used to prevent. 





