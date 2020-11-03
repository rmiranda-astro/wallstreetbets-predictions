import pandas as pd
import requests
import json
import time
import datetime
import pytz


def api_call(url, pause=1.0, max_tries=3):

    """
    Make API request at 'url' and unpack response. In case of a bad response, make
    several additional attempts, waiting 'pause' seconds between requests, in order to
    avoid API limitations. Gives up after 'max_tries' failed attempts and returns an
    empty list. If successful, returns a list of dictionaries representing
    submissions/comments.
    """

    success = False
    tries = 0

    while not success:

        try:

            r = requests.get(url)
            data = json.loads(r.text)["data"]
            success = True

        except:

            tries += 1

            if tries == max_tries:

                return []

            time.sleep(pause)

    return data


def scrape_comments(
    start_date, end_date, subreddit, title, min_comments=100, pause=1.0
):

    """
    Scrapes Reddit comments from threads with title matching 'title' in 'subreddit',
    made on weekdays (M-F) between 'start date' and 'end date' using the reddit
    Pushshift API. Matching threads are ignored if the number of comments is smaller
    than 'min_comments'. The maximum allowed number of comments (1000) are retrieved per
    API call, and the calls are made 'pause' seconds apart to avoid API limitations.
    Returns a pandas dataframe containing the date, time (Unix timestamp), and body of
    the retrieved comments.
    """

    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    df = []

    for date in dates:

        # Initialize 'start' and 'end' timestamps to midnight ET on 'date' and midnight
        # ET on the next day. These specify the timeframe in which to search for
        # matching submissions/comments.

        start = int(date.replace(tzinfo=pytz.timezone("America/New_York")).timestamp())

        end = start + 86400

        # Find matching submissions

        url = (
            "https://api.pushshift.io/reddit/search/submission/"
            "?subreddit=%s&title=%s&after=%s&before=%s&limit=1000"
            % (subreddit, title, str(start), str(end))
        )

        submissions = api_call(url)

        time.sleep(pause)

        for submission in submissions:

            if submission["num_comments"] < min_comments:

                continue

            print("Title:", submission["title"])
            print("Total Comments:", submission["num_comments"])
            print("Downloading Comments...")

            comments = []

            # Retrieve comments from submission, 1000 at a time. Comments are returned
            # chronologically by the API. After each batch of comments, set 'start'
            # equal to the timestamp of the last retrieved comment, so that on the next
            # request we only get comments made after the ones we already have. Continue
            # until we have all comments made on 'date' (in ET).

            while start < end:

                url = (
                    "https://api.pushshift.io/reddit/search/comment/"
                    "?link_id=%s&after=%s&limit=1000" % (submission["id"], start)
                )

                new_comments = api_call(url)

                if len(new_comments) == 0:

                    break

                for comment in new_comments:

                    if comment["created_utc"] < end:

                        comments.append(comment)

                start = new_comments[-1]["created_utc"]

                print(len(comments), end="...", flush=True)

                time.sleep(pause)

            # Append all of the comments retrieved from the current submission to the
            # main list of comments.

            for comment in comments:

                df.append([date, comment["created_utc"], comment["body"]])

            print("\n")

    # Convert the list of comments into a dataframe and rename the columns

    df = pd.DataFrame(df)
    df.columns = ["Date", "Time", "Body"]

    return df


if __name__ == "__main__":

    # Scrape comments from the r/wallstreetbets Daily Discussion threads between
    # September 30, 2019 and September 30, 2020

    df = scrape_comments(
        datetime.date(2019, 9, 30),
        datetime.date(2020, 9, 30),
        "wallstreetbets",
        "daily+discussion+thread",
    )

    df.to_csv("comments_2019-09-30_2020-09-30_new.csv")
