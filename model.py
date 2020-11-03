import numpy as np
import pandas as pd
from sklearn import feature_extraction, metrics, neural_network
import time
import datetime
import pytz

import warnings

# Ignore an unimportant warning raised by numpy when pandas 'read_csv' is called with
# the index column specified.

warnings.simplefilter(action="ignore", category=FutureWarning)


def merge_data(
    comments_file,
    ticker_file,
    filter_symbol=None,
    merge_comments=True,
    chunk_size=None,
    verbose=False,
):

    """
    Read csv files containing comments ('comments_file', created by 'scrape_comments')
    and ticker data ('ticker_file', created by 'get_ticker_data') into dataframes, do
    some pre-processing, and merge them into a final dataframe to be used in a machine
    learning model. The final dataframe has three columns:

    Date: The date on which a comment/chunk of comments was posted.
    Body: The body/text of the comment/chunk of comments.
    CloseUpMove: A binary indicator of whether or not the ticker closing price on the
        trading day after 'Date' is higher than it was on 'Date'.

    'Body' will serve as our "X" data, and 'CloseUpMove' will serve as our "y" data.
    That is, we wish to predict whether tomorrow's closing price will be higher than
    today's closing price, using comments posted today.

    Parameters:

    'filter_symbol': If not None, then only select comments containing the specified
        ticker symbol (e.g., 'AAPL'; case insensitive).

    'merge_comments': If True, then comments made on the same date are merged into
        "chunks" of comments.

    'chunk_size': If 'merge_comments' is True, this specifies the size (i.e., number of
        comments from the same date) of the chunks created by merging comments. If
        'chunk_size' is None, then all of the comments posted on the same date are
        merged into a single chunk.

    'verbose': If True, print some simple diagnostic information about the data.
    """

    df_comments = pd.read_csv(comments_file, index_col=0)

    # Drop deleted/removed comments

    df_comments.dropna(inplace=True)
    df_comments = df_comments[df_comments.Body != "[deleted]"]
    df_comments = df_comments[df_comments.Body != "[removed]"]

    # Drop comments made after 4 PM ET (end of trading day)

    df_comments["Midnight"] = (
        pd.to_datetime(df_comments.Date)
        .dt.tz_localize("America/New_York")
        .dt.tz_convert("GMT")
        .astype(int)
        // 10 ** 9
    )
    df_comments = df_comments[df_comments.Time - df_comments.Midnight < 57600]

    if verbose:
        print("Total Comments:", len(df_comments))

    if filter_symbol != None:

        df_comments = df_comments[
            df_comments.Body.str.contains(filter_symbol, case=False, na=False)
        ]

        if verbose:
            print("Comments Containing Symbol:", len(df_comments))

    if merge_comments and chunk_size == None:

        # Gather all of the comments made on the same date together and merge them into
        # a chunk (by joining them with a line break)

        df_comments_merged = (
            df_comments.groupby("Date")["Body"].apply("\n".join).reset_index()
        )

    elif merge_comments and chunk_size != None:

        # Join comments posted on the same date into chunks of 'chunk_size' comments.
        # The remainder of comments, after being divided into as many chunks as
        # possible, are dropped (i.e., there are no partial chunks).

        comments_merged = []

        df_comment_lists = df_comments.groupby("Date")["Body"].apply(list).reset_index()

        for idx, row in df_comment_lists.iterrows():

            date = row.Date
            comment_list = row.Body

            split = [i * chunk_size for i in range(len(comment_list) // chunk_size + 1)]

            chunks = [
                comment_list[split[i] : split[i + 1]] for i in range(len(split) - 1)
            ]
            chunks += [comment_list[split[-1] :]]

            if len(chunks[-1]) != chunk_size:
                chunks.pop()

            if len(chunks) == 0:
                continue

            comments_merged += [[date, "\n".join(chunk)] for chunk in chunks]

        df_comments_merged = pd.DataFrame(comments_merged, columns=["Date", "Body"])

    # Define the 'CloseUpMove' column for the ticker data, which is equal to 1 if the
    # closing price for the following day is higher than the closing price for the
    # current day, and 0 otherwise.

    df_ticker = pd.read_csv(ticker_file, index_col=0)
    df_ticker["CloseUpMove"] = (df_ticker.Close.shift(-1) - df_ticker.Close > 0).astype(
        int
    )
    df_ticker = df_ticker[:-1]

    # Join the comment data (comments or chunks of comments) and the ticker data on the
    # Date column, i.e., create a dataframe with a row for each comment/chunk of
    # comments with ticker data for its date.

    if merge_comments:

        df = pd.merge(df_comments_merged, df_ticker, on="Date", how="inner")

    else:

        df = pd.merge(df_comments, df_ticker, on="Date", how="inner")

    # Select only the three columns of interest.

    df = df[["Date", "Body", "CloseUpMove"]]

    if verbose:

        print("Number of Samples:", len(df))
        print("Unique Dates:", len(df.Date.unique()))

    return df


def train_dev_test_split(df, split_dates):

    """
    Split a dataframe (produced by 'merge_data') into chronologically sequential
    training, development, and test sets, using the dates specified in 'split_dates'.
    This is a list of four dates (YYYY-MM-DD strings), where the first three indicate
    the start dates for the training, development, and test sets, and the fourth
    indicates the end date for the test set. Return the three sets as dataframes.
    """

    df_train = df[(df.Date >= split_dates[0]) & (df.Date < split_dates[1])]
    df_dev = df[(df.Date >= split_dates[1]) & (df.Date < split_dates[2])]
    df_test = df[(df.Date >= split_dates[2]) & (df.Date <= split_dates[3])]

    return df_train, df_dev, df_test


def initialize_vect(num_features, max_df=0.5):

    """
    Initialize and return the text vectorizer for our model. This is just a shortcut for
    initializing sklearn's tf-idf vectorizer with some default parameters. We tokenize
    on words made up of at least two letters, use sklearn's default stop words with some
    additions to filter out urls, and use both monograms and bigrams.

    The number of tf-idf features to be used is specified by the argument
    'num_features'. A maximum document frequency for features ('max_df') to be used
    instead of the default value (0.5) can also be specified.
    """

    new_sw = [
        "amp",
        "com",
        "html",
        "http",
        "https",
        "reddit",
        "wallstreetbets",
        "watch",
        "www",
        "youtu",
        "youtube",
    ]

    stop_words = feature_extraction.text.ENGLISH_STOP_WORDS.union(new_sw)

    vect = feature_extraction.text.TfidfVectorizer(
        token_pattern="(?ui)\\b\\w*[a-z]\\w*[a-z]+\\b",
        ngram_range=(1, 2),
        stop_words=stop_words,
        max_df=max_df,
        max_features=num_features,
    )

    return vect


def initialize_nn(layer_size, alpha):

    """
    Initialize an artificial neural network classifier with some default parameters
    (tanh activation function and a random seed for reproducibility). The number of
    hidden units in the (single) hidden layer 'layer size' and the regularization
    parameter 'alpha' need to be specified.
    """

    clf = neural_network.MLPClassifier(
        hidden_layer_sizes=(layer_size),
        activation="tanh",
        alpha=alpha,
        max_iter=2000,
        random_state=99,
    )

    return clf


def get_predictions(df, vect, clf, thresh=0.5):

    """
    Get predictions from a trained model (vectorizer 'vect' and classifier 'clf') using
    the inputs (comments/chunks of comments) in a data set (dataframe 'df'). The model
    is trained on/makes predictions based on multiple samples (chunks of comments) per
    date. Since we only want a single prediction per date, we shouldn't treat these as
    independent predictions. Instead we treat the different predictions for the same
    date as "votes" for the overall prediction for that date.

    The parameter 'thresh' specifies the fraction of positive predictions needed to make
    a positive prediction for a given date, which is 50% by default, but in principle
    could be tuned to improve model performance.

    The predicted y-values, as well as the true y-values, for the unique dates in 'df'
    are returned as lists.
    """

    dates = df.Date.unique()

    y_pred, y_true = [], []

    for date in dates:

        d = df[df.Date == date].reset_index(drop=True)
        X = vect.transform(d.Body)
        y = clf.predict(X)

        y_pred.append(int(np.mean(y) >= thresh))
        y_true.append(int(d.CloseUpMove.iloc[0]))

    return y_pred, y_true


def score_model(df_score, vect, clf, verbose=False):

    """
    Evaluate the performance of a trained model (vectorizer 'vect' and classifier 'clf')
    on a data set (dataframe 'df_score'). We first get the predicted y-values (as well
    as the true y-values) using 'get_predictions', and then calculate and return the
    metrics of interest (the accuracy and F1 score for the model/data set).

    Parameters:

    'verbose': If True, print out the predicted and true y-values.
    """

    y_pred, y_true = get_predictions(df_score, vect, clf)

    if verbose:

        print("Pred:", y_pred)
        print("True:", y_true)

    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)

    return acc, f1


def score_dummy_model(df_score):

    """
    Evaluate the performance of a dummy classifier which always returns 1 for y. Returns
    the accuracy and F1 score for this model on the data set 'df_score'.
    """

    dates = df_score.Date.unique()

    y_true = []

    for date in dates:

        d = df_score[df_score.Date == date].reset_index(drop=True)
        y_true.append(int(d.CloseUpMove.iloc[0]))

    acc = metrics.accuracy_score(y_true, np.ones(len(y_true)))
    f1 = metrics.f1_score(y_true, np.ones(len(y_true)))

    return acc, f1


def nn_parameter_search(
    comments_file, ticker_file, split_dates, search_params, verbose=False
):

    """
    Perform a parameter search for evaluating the performance of different artificial
    neural network models. Takes csv files containing comments and ticker data
    ('comments_file' and 'ticker_file') and a list of dates for the split into training,
    development, and test sets ('split_dates'), trains different models on the training
    set, then reports their performance on the training and development sets.

    The different models are specified by 'search_params', a dictionary containing lists
    of values for four parameters: 'Chunk Size' (the number of comments in a
    chunk/training example), 'Num Features' (the number of features used by the text
    vectorizer), 'Layer Size' (the number of hidden units in the single hidden layer of
    the neural network) and 'Alpha' (the regularization parameter used by the neural
    network). Each key in the dictionary is a parameter name, and its value is a list of
    numerical values for that parameter.

    Return a list of results for each model (i.e., set of parameter values). Each result
    is a dictionary, giving the value for each parameter, as well as the metrics
    (accuracy and F1 score) for the training and development sets for the model. Metrics
    for a dummy classifier are also given for comparison.

    Additional parameters:

    'verbose': If True, print out the results for the perfomance of each model.
    """

    results = []

    for chunk_size_val in search_params["Chunk Size"]:

        # Create a dataframe with comments merged as specified by 'chunk_size'.

        df = merge_data(comments_file, ticker_file, chunk_size=chunk_size_val)
        df_train, df_dev, df_test = train_dev_test_split(df, split_dates)

        # Evaluate the performance of a dummy classifier on the training and development
        # sets. This needs to be done for each value of 'chunk_size' (for large values
        # of 'chunk_size', the number of samples in the data set may be reduced since
        # dates for which we have fewer than 'chunk_size' comments may be dropped).

        acc_train_dum, f1_train_dum = score_dummy_model(df_train)
        acc_dev_dum, f1_dev_dum = score_dummy_model(df_dev)

        result_dum = {
            "Chunk Size": chunk_size_val,
            "Dummy Accuracy (Train)": round(acc_train_dum, 4),
            "Dummy Accuracy (Dev)": round(acc_dev_dum, 4),
            "Dummy F1 Score (Train)": round(f1_train_dum, 4),
            "Dummy F1 Score (Dev)": round(f1_dev_dum, 4),
        }

        if verbose:

            print(result_dum)

        results.append(result_dum)

        for num_features_val in search_params["Num Features"]:

            vect = initialize_vect(num_features=num_features_val)

            X_train = vect.fit_transform(df_train.Body)
            y_train = df_train.CloseUpMove

            for layer_size_val in search_params["Layer Size"]:

                for alpha_val in search_params["Alpha"]:

                    clf = initialize_nn(layer_size_val, alpha_val)
                    clf.fit(X_train, y_train)

                    # Evaluate the performance of the model on the training and
                    # development sets.

                    acc_train, f1_train = score_model(df_train, vect, clf)
                    acc_dev, f1_dev = score_model(df_dev, vect, clf)

                    result = {
                        "Chunk Size": chunk_size_val,
                        "Num Features": num_features_val,
                        "Hidden Layers": layer_size_val,
                        "Alpha": alpha_val,
                        "Accuracy (Train)": round(acc_train, 4),
                        "Accuracy (Dev)": round(acc_dev, 4),
                        "F1 Score (Train)": round(f1_train, 4),
                        "F1 Score (Dev)": round(f1_dev, 4),
                    }

                    if verbose:

                        print(result)

                    results.append(result)

    return results


def test_final_model(comments_file, ticker_file, split_dates, params, verbose=False):

    """
    Test a final model (i.e., with parameters chosen after optimizing performance on the
    development set, using 'nn_parameter_search') on the unseen test set. The comment
    file, ticker file, and split dates should be the same ones used in
    'nn_parameter_search'. The parameters are specified by the dictionary 'params', in
    which the keys are the parameter names ('Chunk Size', 'Num Features',
    'Hidden Layers', and 'Alpha') and the keys are (single) values for each parameter.
    The model is trained on the training set and the performance metrics (accuracy and
    F1 score) for the model on the test set, as well as the same metrics for a dummy
    classifier, are returned as a dictionary.

    Additional parameters:

    'verbose': If True, print out the metrics and predictions of the model.
    """

    df = merge_data(comments_file, ticker_file, chunk_size=params["Chunk Size"])
    df_train, df_dev, df_test = train_dev_test_split(df, split_dates)

    vect = initialize_vect(num_features=params["Num Features"])

    X_train = vect.fit_transform(df_train.Body)
    y_train = df_train.CloseUpMove

    clf = initialize_nn(params["Hidden Layers"], params["Alpha"])
    clf.fit(X_train, y_train)

    acc, f1 = score_model(df_test, vect, clf, verbose=verbose)
    acc_dum, f1_dum = score_dummy_model(df_test)

    result = {
        "Accuracy": round(acc, 4),
        "F1 Score": round(f1, 4),
        "Dummy Accuracy": round(acc_dum, 4),
        "Dummy F1 Score": round(f1_dum, 4),
    }

    if verbose:

        print("Accuracy: %.4f (Dummy Accuracy: %.4f)" % (acc, acc_dum))
        print("F1 Score: %.4f (Dummy F1 Score: %.4f)" % (f1, f1_dum))

    return result


if __name__ == "__main__":

    # Specify a csv file containing comments (created by 'scrape_comments') and a csv
    # file containing ticker data (created by 'get_ticker_data'), and the dates to be
    # used for splitting the data into train, development, and test sets (see
    # 'train_dev_test_split').

    comments_file = "comments_2019-09-30_2020-09-30.csv"
    ticker_file = "spy.csv"
    split_dates = ["2019-09-30", "2020-07-01", "2020-08-16", "2020-09-30"]

    # Perform a parameter search for evaluating different models (by training on the
    # training set and making predictions on the development set):

    search_params = {
        "Chunk Size": [256, 512, 1024],
        "Num Features": [1024, 2048, 4096],
        "Layer Size": [64, 128, 256],
        "Alpha": np.logspace(-4.0, 0.0, 5),
    }

    search_results = nn_parameter_search(
        comments_file, ticker_file, split_dates, search_params, verbose=True
    )

    # Test a model (on the unseen test set) for a particular set of parameters:

    test_params = {
        "Chunk Size": 256,
        "Num Features": 2048,
        "Hidden Layers": 64,
        "Alpha": 0.001,
    }

    test_result = test_final_model(
        comments_file, ticker_file, split_dates, test_params, verbose=True
    )
