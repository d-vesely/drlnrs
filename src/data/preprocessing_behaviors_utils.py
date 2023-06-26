import os
import pandas as pd
from tqdm.auto import tqdm

tqdm.pandas()


def read_data_behaviors(path):
    """Read behaviors data into pandas dataframe"""
    # Define columns
    cols_behaviors = [
        "id",
        "user_id",
        "time",
        "history",
        "impression"
    ]
    # Read data from .tsv-file
    data_behaviors = pd.read_csv(
        os.path.join(path, "behaviors.tsv"),
        sep="\t",
        header=None,
        names=cols_behaviors
    )
    return data_behaviors


def concat_data_behaviors(data_behaviors_list, save_dir=None):
    """Concatenate behaviors dataframes

    Arguments:
        data_behaviors_list -- list of behaviors dataframes

    Keyword Arguments:
        save_dir -- where to save the concatenated data. If None, data
        will not be saved (default: {None})

    Returns:
        concatenated dataframe
    """
    # Concatenate
    data_behaviors = pd.concat(data_behaviors_list)

    # Pickle data, is save_dir provided
    if save_dir is not None:
        data_behaviors.to_pickle(
            os.path.join(
                save_dir,
                "behaviors_concat.pkl"
            )
        )

    return data_behaviors


def separate_impression_col(impression):
    """Split impression into lists of shown, clicked and ignored news

    Arguments:
        impression -- list of space-separated news IDs with a label. ID and label are separated by a hyphen. Label is 1 if news was clicked, 0 if news was ignored

    Raises:
        ValueError: the label was unexpected

    Returns:
        list of shown, clicked and ignored news IDs
    """
    shown_news = []
    clicked_news = []
    ignored_news = []
    # Split into individual impressions
    impressions = impression.split(" ")
    for imp in impressions:
        # Split news id and label
        news_id, action = imp.split("-")
        shown_news.append(news_id)
        if action == "1":
            clicked_news.append(news_id)
        elif action == "0":
            ignored_news.append(news_id)
        else:
            # Unexpected label
            raise ValueError("Invalid action found.")
    return shown_news, clicked_news, ignored_news


def process_user_histories(data_behaviors, build_ignore_history=False):
    """Amend user click histories with provided data

    Click histories in the data remain unchanged within a split. We incrementally
    amend the histories with the click information in the split itself.

    Arguments:
        data_behaviors -- the behaviors dataframe, must be sorted by user ID (primary)
        and timestamp (secondary)

    Returns:
        data_behaviors: the behaviors dataframe with amended histories
        data_users: a dataframe mapping initial click histories to user IDs
    """
    initial_user_histories = {}
    amended_histories = []

    prev_user = None
    prev_clicked_news = None

    if build_ignore_history:
        amended_ignore_histories = []
        prev_ignored_news = None

    #! Iterate over data behaviors
    #! The dataframe is expected to be sorted by user ID (primary)
    #! and timestamp (secondary)
    for row in tqdm(data_behaviors.itertuples(), total=data_behaviors.shape[0]):
        # Get user and list of clicked news from sample
        curr_user = row.user_id
        curr_clicked_news = row.clicked_news
        if build_ignore_history:
            curr_ignored_news = row.ignored_news

        # Same user
        if curr_user == prev_user:
            # New history is old history extended by clicked news from previous sample
            curr_history = amended_histories[-1].copy()
            curr_history.extend(prev_clicked_news)
            if build_ignore_history:
                curr_ignore_history = amended_ignore_histories[-1].copy()
                curr_ignore_history.extend(prev_ignored_news)

        # New user
        else:
            # Get initial click history and save it
            curr_history = row.history
            if build_ignore_history:
                curr_ignore_history = []
            initial_user_histories[curr_user] = curr_history

        amended_histories.append(curr_history)

        prev_user = curr_user
        prev_clicked_news = curr_clicked_news

        if build_ignore_history:
            amended_ignore_histories.append(curr_ignore_history)
            prev_ignored_news = curr_ignored_news

    # Replace history column with list of amended histories
    data_behaviors["history"] = amended_histories
    if build_ignore_history:
        data_behaviors["ignore_history"] = amended_ignore_histories

    # Create users dataframe containing initial histories
    data_users = pd.DataFrame(
        initial_user_histories.items(),
        columns=["user_id", "history"]
    )

    return data_behaviors, data_users


def time_of_day_map(hour):
    """Map numeric hour to a time-of-day category"""
    time_of_day = "nighttime"
    if 5 <= hour < 11:
        time_of_day = "morning"
    if 11 <= hour < 17:
        time_of_day = "daytime"
    elif 17 <= hour < 23:
        time_of_day = "evening"
    return time_of_day


def preprocess_data_behaviors(data_behaviors, save_dir=None, save_name_suffix="", test_data=False,
                              exploration=True, get_relevant_news=True,
                              save_user_histories=False, build_ignore_history=False):
    """Preprocessing routine for behaviors data

    Arguments:
        data_behaviors -- behaviors data as pandas dataframe

    Keyword Arguments:
        save_dir -- where to save the data. If None, data 
        will not be saved (default: {None})
        test_data -- whether the behaviors data is from the test split
        exploration -- whether to construct columns for exploration purposes
        get_relevant_news -- whether to get set of occurring news IDs
        save_user_histories -- whether to save (redundant) initial user histories

    Raises:
        ValueError: when both test_data and exploration are set to True

    Returns:
        a dict of dataframes containing various preprocessed data
    """
    if test_data:
        if exploration:
            raise ValueError(
                "Test data cannot be preprocessed in exploration mode."
            )
        if save_user_histories:
            raise ValueError(
                "User histories are not collected for test data."
            )

    # Construct save_dir, if provided
    if save_dir is not None:
        # Create "preprocessed" directory within save_dir
        save_dir = os.path.join(save_dir, "preprocessed")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"[INFO] preprocessed data will be saved in {save_dir}")
    else:
        print("[WARN] preprocessed data will not be saved")

    # Convert time column to datetime timestamp
    print("[INFO] converting timestamp data")
    # 11/15/2019 9:55:12 AM
    timestamp_format = "%m/%d/%Y %I:%M:%S %p"
    data_behaviors["timestamp"] = pd.to_datetime(
        data_behaviors["time"],
        format=timestamp_format
    )

    # Split history string into list of news IDs
    print("[INFO] splitting reading history")
    data_behaviors["history"] = data_behaviors["history"].str.split(" ")
    # Properly handle empty histories
    data_behaviors["history"] = data_behaviors["history"].apply(
        lambda h: h if isinstance(h, list) else []
    )

    if test_data:
        # In test data the impressions are not labeled
        print("[INFO] splitting impression column")
        data_behaviors["shown_news"] = data_behaviors["impression"].str.split(
            " "
        )
        data_behaviors["shown_news"] = data_behaviors["shown_news"].apply(
            lambda h: h if isinstance(h, list) else []
        )

    else:
        # Split impressions into lists of news IDs
        print("[INFO] separating impression column")
        shown_news, clicked_news, ignored_news = zip(
            *data_behaviors["impression"].progress_apply(separate_impression_col)
        )
        # Put each list in own column
        data_behaviors["clicked_news"] = clicked_news
        data_behaviors["ignored_news"] = ignored_news
        data_behaviors["shown_news"] = shown_news

        # Sort by user ID (primary) and timestamp (secondary)
        # This is required by process_user_histories
        print("[INFO] sorting data")
        data_behaviors.sort_values(["user_id", "timestamp"], ascending=[
            True, True], inplace=True)

        print("[INFO] processing user histories")
        data_behaviors, data_users = process_user_histories(
            data_behaviors, build_ignore_history=build_ignore_history)

    # Drop redundant columns
    data_behaviors.drop(
        labels=["impression", "time"],
        axis=1,
        inplace=True
    )

    # Pickle data, if save_dir provided
    if save_dir is not None:
        print("[INFO] saving preprocessed data")
        data_behaviors.to_pickle(os.path.join(
            save_dir, f"behaviors{save_name_suffix}.pkl"))

        # Pickle initial user histories (redundant), if wanted
        if save_user_histories:
            data_users.to_pickle(os.path.join(save_dir, "users.pkl"))

    # Collect additional exploratory data, if wanted
    if exploration:
        print("[INFO] obtaining exploration data")
        # Get initial history lengths from users data
        data_users["history_length"] = data_users["history"].str.len()

        # Get numbers of clicked/ignored/shown news
        news_count_cols = ["clicked_news", "ignored_news", "shown_news"]
        for col in news_count_cols:
            data_behaviors[f"{col}_length"] = data_behaviors[col].str.len()

        # Get percentages of clicked/ignored news
        data_behaviors["clicked_news_percent"] = 100 * \
            data_behaviors["clicked_news_length"] / \
            data_behaviors["shown_news_length"]
        data_behaviors["ignored_news_percent"] = 100 * \
            data_behaviors["ignored_news_length"] / \
            data_behaviors["shown_news_length"]

        # Extract date, time, hour and time-of-day-category from timestamp
        data_behaviors["ts_date"] = data_behaviors["timestamp"].dt.strftime(
            "%Y-%m-%d"
        )
        data_behaviors["ts_time"] = data_behaviors["timestamp"].dt.strftime(
            "%H:%M:%S"
        )
        data_behaviors["ts_hour"] = data_behaviors["timestamp"].dt.hour
        data_behaviors["time_category"] = data_behaviors["ts_hour"].map(
            time_of_day_map
        )

        # Get stats for numeric columns
        columns = [
            "clicked_news_length",
            "ignored_news_length",
            "shown_news_length",
            "clicked_news_percent",
            "ignored_news_percent"
        ]
        # List of percentiles to get stats for
        percentiles = [0.25, 0.5, 0.75, 0.95, 0.99]
        data_stats = get_numeric_value_stats(
            data_behaviors,
            columns,
            percentiles
        )

        # Pickle data, if save_dir provided
        if save_dir is not None:
            print("[INFO] saving exploratory data")
            # Store exploration data with 'exp_' prefix filename
            data_behaviors.to_pickle(
                os.path.join(save_dir, "exp_behaviors.pkl")
            )
            data_stats.to_pickle(
                os.path.join(save_dir, "exp_stats.pkl")
            )

            # Pickle initial user histories (redundant), if wanted
            if save_user_histories:
                data_users.to_pickle(os.path.join(save_dir, "exp_users.pkl"))

    # Collect all unique news IDs from the behaviors data, if wanted
    if get_relevant_news:
        print("[INFO] collecting set of relevant news IDs")
        if test_data:
            data_relevant_news = get_relevant_news_set(data_behaviors)
        else:
            data_relevant_news = get_relevant_news_set(
                data_behaviors, data_users)

        # Pickle data, if save_dir provided
        if save_dir is not None:
            print("[INFO] saving relevant news data")
            data_relevant_news.to_pickle(
                os.path.join(save_dir, "relevant_news.pkl")
            )

    # Build list of dataframes to return
    returns = {"behaviors": data_behaviors}
    if not test_data:
        returns["users"] = data_users
    if exploration:
        returns["stats"] = data_stats
    if get_relevant_news:
        returns["relevant_news"] = data_relevant_news

    return returns


def get_survival_data(data_behaviors):
    """Get how long each news remained relevant

    For each news ID in the data, this function finds the timestamp of
    the first occurrence and the last occurrence. The delta between the two
    is the "survival time" of the news.

    Arguments:
        data_behaviors -- preprocessed behaviors data as pandas dataframe

    Returns:
        dict containing survival data, keys are news IDs
    """
    survival_data = {}
    # Sort data by time
    data_behaviors.sort_values(by="timestamp", inplace=True)
    for row in tqdm(data_behaviors.itertuples(), total=data_behaviors.shape[0]):
        shown_news = row.shown_news
        timestamp = row.timestamp
        for news_id in shown_news:
            if news_id not in survival_data:
                # Unseen news ID
                survival_data[news_id] = {
                    "first_impression": timestamp
                }
            # This COULD be the last impression
            # These calculations might be redundant, but they are quick
            # If news ID occurs only once, first and last impression are equal
            # and delta is 0
            survival_data[news_id]["last_impression"] = timestamp
            delta = timestamp - survival_data[news_id]["first_impression"]
            survival_time_hrs = delta.total_seconds() / 3600
            survival_data[news_id]["survival_time_hrs"] = survival_time_hrs

    return survival_data


def get_engagement_data(data_behaviors):
    """Collect engagement data for each news

    For each news ID in the data, this function collects how often it
    was shown/clicked/ignored.

    Arguments:
        data_behaviors -- preprocessed behaviors data as pandas dataframe

    Returns:
        dict containing engagement data, keys are news IDs
    """
    engagement_data = {}
    for row in tqdm(data_behaviors.itertuples(), total=data_behaviors.shape[0]):
        shown_news = row.shown_news
        clicked_news = row.clicked_news
        for news_id in shown_news:
            if news_id not in engagement_data:
                # Unseen news ID
                engagement_data[news_id] = {
                    "shown": 0,
                    "ignored": 0,
                    "clicked": 0
                }
            engagement_data[news_id]["shown"] += 1
            # Ignored news are the remaining news that were not clicked
            # Checking if shown news were clicked is fast, usually only few
            # items are clicked
            if news_id in clicked_news:
                engagement_data[news_id]["clicked"] += 1
            else:
                engagement_data[news_id]["ignored"] += 1

    return engagement_data


def get_relevant_news_set(data_behaviors, data_users=None):
    """Collect set of news IDs that are in the data

    For each news ID in the data, this method records whether the news occurs 
    in a user history, or in an impression.

    Arguments:
        data_behaviors -- preprocessed behaviors data as pandas dataframe
        data_users -- initial user histories data as pandas dataframe

    Returns:
        dataframe containing all relevant news IDs and occurrence information
    """
    # Build sets
    shown_news = set()
    history_news = set()
    for news in data_behaviors["shown_news"]:
        shown_news.update(news)
    # Only collect IDs from initial histories
    if data_users is not None:
        for news in data_users["history"]:
            history_news.update(news)
    else:
        for news in data_behaviors["history"]:
            history_news.update(news)

    # Build list of relevant news IDs
    relevant_news = list(set().union(shown_news, history_news))
    # Create a dictionary to hold the boolean values for each ID
    dict_relevant_news = {
        "news_id": relevant_news,
        "shown": [True if nid in shown_news else False for nid in relevant_news],
        "history": [True if nid in history_news else False for nid in relevant_news]
    }
    # Store as dataframe
    data_relevant_news = pd.DataFrame(dict_relevant_news)
    return data_relevant_news


def get_unique_user_set(data_behaviors):
    """Return the set of unique user IDs in the data"""
    return set(data_behaviors["user_id"].unique())


def get_numeric_value_stats(data_behaviors, columns, percentiles, data_users=None):
    """Collect numeric descriptors for given columns

    Arguments:
        data_behaviors -- preprocessed behaviors data as pandas dataframe
        columns -- which columns to collect stats for
        percentiles -- which percentiles to collect

    Keyword Arguments:
        data_users -- optional preprocessed user data as pandas dataframe. Collect
        stats on history length column. (default: {None})

    Returns:
        Stats in a pandas dataframe
    """
    stats = []
    for col in columns:
        stats.append(data_behaviors[col].describe(percentiles=percentiles))
    if data_users is not None:
        stats.append(data_users["history_length"].describe(
            percentiles=percentiles)
        )
    stats_combined = pd.concat(stats, axis=1)
    return stats_combined


def get_time_data(data_behaviors):
    """Get the first and last timestamp, and the delta, for given behaviors data"""
    min = data_behaviors["timestamp"].min()
    max = data_behaviors["timestamp"].max()
    span = max - min
    time_data = {
        "min": min,
        "max": max,
        "span": span
    }
    return time_data
