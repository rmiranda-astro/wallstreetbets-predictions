# Overview

r/wallstreetbets is an active financial subreddit focused on short-term options trading, featuring heavy discussion of the day-to-day and intraday movements of stocks. The goal of this project is to answer the following question: can we use the comments made on r/wallstreetbets to predict the movement of the stock market? Specifically, can we predict if a stock's closing price will be higher tomorrow than it was today, based on the today's comments? For this experiment we focus on the movement of SPY, an exchange traded funded (ETF) that tracks the S&P 500 index, and hence the overall movement of the stock market. We attempt to model its movement using a natural language processing/machine learning model that takes r/wallstreetbets comments as its input.

Note that the procedure described here is not a sentiment analysis. Although we could try to determine the sentiment of the subreddit and try to use it as a predictor of stock movements, we assume here that doing so would unnecessarily filter out a large amount of information contained in the comments, and limit the predictive power of our model. Instead, we wish to use the comments themselves as the direct input to the model, while remaining agnostic to the sentiment they reflect.

# Data Collection and Preprocessing

We scrape comments from r/wallstreetbets using the Pushshift API for reddit. As a starting point, we restrict ourselves to the daily discussion threads, which typically receive tens of thousands of comments (this number was somewhat smaller prior to Feb/March 2020), and which contain the majority of the comments made on the subreddit during a trading day. For our dataset, we scraped one year's worth of comments from the daily discussion threads (between 09/30/2019 and 09/30/2020), amounting to about 4.5 million comments. We also gathered SPY ticker data for the same range of dates using the yfinance package.

We perform some preprocessing on the raw comment and ticker data to prepare it for our machine learning model. We first filter out deleted and removed comments. We also ignore comments posted after the end of market hours (4 PM ET). If our model is sufficiently predictive, its envisioned use is to make predictions that inform trading decisions made at the end of the trading day (since we make a prediction about how tomorrow's closing price compares to today's closing price), and so we do not want to take comments made after market hours into consideration.

Since we don't expect any individual comment to be have much predictive power, we aggregate comments made on the same day into "chunks" of comments. However, merging all of the comments for one day into a single chunk has the undesirable effect of producing a relatively small number of data samples (one for each trading day). We would prefer to have a large number of samples for more effective model training. Therefore, we use chunks of a few hundred to about a thousand comments. The size of the chunks is a tunable parameter in our model. Larger chunk sizes result in fewer training samples, but may enhance the predictive power of each one. Note that while this trick allows us to get many data samples for of a single day, we do not treat them independently when evaluating the model. Instead, for each day in the data set, we use the predictions of the multiple samples as votes to produce a final prediction for the day.

# The Model

The chunks of comments constitute our "X" data. Our "y" data is the binary indicator of whether or note our stock symbol closes at a higher price on the following trading day, as compared to its closing price on the day the chunk of comments was posted. Our "y" data is categorical (an upward movement is the positive class, and a downward movement is the negative class), making our task a classification problem.

We use a term frequency-inverse document frequency (tf-idf) representation to vectorize the text of the comment chunks so that they can be fed into our classifier, a fully connected artificial neural network with a single hidden layer. We make use use sklearn's implementation of the tf-idf vectorizer and multi-layer perceptron classifier. We vary four parameters in our model: the chunk size (the number of comments that are merged into a chunk), number of features (most frequent words/bigrams) used in the tf-idf vectorization, number of hidden units in the hidden layer of the neural network, and the regularization parameter for the neural network.

The data was split into a training set (09/30/2019 to 06/30/2020; 188 trading days), development set (07/01/2020 to 08/15/2020; 31 trading days), and test set (08/16/2020 to 09/30/2020; 31 trading days).

# Results

We carried out a parameter search to optimize the model performance. We considered values for the chunk size between 256 and 1024, number of tf-idf features between 1024 and 4096, number of hidden layers between 64 and 256, and regularization parameter between 0.0001 and 1. For each of the 135 points in the parameter grid, the model was trained on the training set and its performance was evaluated on the development set. We used the accuracy (fraction of correctly classified examples) and F1 score (harmonic mean of precision and recall) as our performance metrics. We also compared the model performance (using the aforementioned metrics) to that of the simplest possible model: a "dummy" classifier which always predicts the positive class (i.e., an upward movement). The usefulness of our more sophisticated model hinges on its ability to outperform this simple (but surprisingly effective) model.

Regardless of the values of the other parameters, with sufficiently strong regularization (values of alpha between 0.1 and 1), the predictions of the model converge towards those of the dummy classifier, resulting in an accuracy of 67.7% and an F1 score of 80.8% on the development set. In choosing our optimal model, we look for models that don't behave as dummy classifiers. Although this criterion results in worse performance on the development set than a dummy classifier, it places importance on the flexibility of the model, i.e., its ability to make more varied predictions.

As our second criterion, we look for models which do not overfit the training set, i.e., those for which the difference in performance on the training and development sets is not too severe. More complex models (i.e., those with a larger chunk size, more tf-idf features, or more hidden units) with weak regularization tend overfit the training set, producing perfect or nearly perfect performance on the training set, but poorly generalizing to the development set.

With these considerations in mind, we find that a similar level performance is achieved by a few different models. We choose from these by choosing the simplest (smaller chunks, fewer features and hidden layers) among them. We settle on a final model, whose parameters and performance are summarized below.

|Chunk Size | Num Features | Hidden Units | Alpha | Accuracy (Train/Dev/Test) | F1 Score (Train/Dev/Test) |
|-----------|--------------|--------------|-------|---------------------------|---------------------------|
|   256     |     2048     |     64       | 0.001 |   91.5% / 64.5% / 67.7%   |  93.3% / 78.4% / 79.2%    |

Surprisingly, the performance of our final model on the test set (accuracy 67.7% and F1 score 79.2%) is slightly better than that of the dummy classifier (accuracy 64.5% and F1 score 78.4%). This difference in performance is just the result of the correct classification of one additional sample (day), as compared to the dummy classifier. It is nonetheless interesting, as it shows that our model has some potential to be useful. The full predictions for the 31 days in the test set are as follows.

|         | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---------|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|Predicted|+|+|+|+|+|-|+|+|+|+|+|+|+|+|+|+|+|+|+|+|-|-|+|+|+|+|+|+|+|+|+|
|   True  |+|-|+|+|+|+|+|+|+|-|+|+|-|-|-|+|-|+|+|+|-|-|-|-|+|+|+|+|-|+|+|

Clearly the model predicts too many positive classes, but it does have the flexibility to predict negative classes, and is able to classify several of them correctly.

# Code

* [scrape_comments.py](https://github.com/rmiranda-astro/wallstreetbets-predictions/blob/main/scrape_comments.py): Scrape comments from reddit using the Pushshift API.  Collects comments from threads matching a given title string for a specified subreddit and range of dates.

* [get_ticker_data](https://github.com/rmiranda-astro/wallstreetbets-predictions/blob/main/get_ticker_data.py): Fetches daily ticker data for a specified ticker symbol and date range.

* [model.py](https://github.com/rmiranda-astro/wallstreetbets-predictions/blob/main/model.py): Contains functions for data pre-processing and for training, optimizing, and making predictions using the machine learning model described above.

See the docstrings of the individual functions in these files for more details on their functionality.

# Future Directions

* Use a larger body of comments, e.g., use comments from other threads besides the wallstreetbets daily discussions, to improve the performance of the model. This requires some modification to the comment scraping function.

* Test the model on other ticker symbols. The performance of the model may be better or worse on other stocks/ETFs besides SPY. This can easily be explored using the existing code, by simply collecting the ticker data for a different symbol, and training/optimizing the model as we have done here. We may also wish to modify the model to predict the movements of multiple stocks at once.

* Feature engineering: in the current implementation, the features are the tf-idf values for the most frequent words and bigrams in the comment training set (no stemming/lemmatization is used, in a limited exploration we found these to have almost no effect on the model performance). Currently we ignore numbers (i.e., "words" containing only numerical digits and no letters) when selecting features. These numbers often specify the target or predicted stock prices of commenters. By comparing these target/predicted prices to the price at the time of the comment, we could produce additional features to be incorporated into the model, which may improve its performance. In principle there are many other features that could be constructed from the comment text.

* Time series analysis: rather a simple day-to-day prediction, we could try to predict changes in the "momentum" of a stock's movement.
