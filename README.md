# **Predicting Stocks with Financial News with LSTM, Differential Privacy and Sentiment Analysis**


---

Team member: Shamal Lalvani and Yunan Wu



# **1.   Introduction**

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



**2.   List item**







