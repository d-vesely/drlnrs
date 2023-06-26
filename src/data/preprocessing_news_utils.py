import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import pandas as pd
import re
from tqdm.auto import tqdm

from .category_maps import CATEGORY_MAP, SUB_CATEGORY_MAP
from .common_utils import read_pickled_data
from .preprocessing_behaviors_utils import get_survival_data, get_engagement_data

nltk.download("punkt")
nltk.download("stopwords")
tqdm.pandas()

# Map from contractions to cleaned up words
CONTRACTIONS = {
    "n't": "not",
    "'s": "is",
    "'ll": "will",
    "'m": "am",
    "'ve": "have"
}


def read_data_news(path):
    """Read news data into pandas dataframe"""
    # Define columns
    cols_news = [
        "news_id",
        "category",
        "sub_category",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities"
    ]
    # Read data from .tsv-file
    data_news = pd.read_csv(
        os.path.join(path, "news.tsv"),
        sep="\t",
        header=None,
        names=cols_news
    )
    return data_news


def concat_data_news(data_news_list, save_dir=None):
    """Concatenate news dataframes"""
    # Concatenate, drop duplicates and re-index
    data_news = pd.concat(data_news_list)
    data_news.drop_duplicates(inplace=True)
    data_news.reset_index(drop=True, inplace=True)

    # Pickle data, if save_dir provided
    if save_dir is not None:
        data_news.to_pickle(os.path.join(save_dir, "news_concat.pkl"))

    return data_news


def _remap_categories(data_news):
    data_news.replace({"category": CATEGORY_MAP}, inplace=True)
    for c, sc_map in SUB_CATEGORY_MAP.items():
        mask = data_news["category"] == c
        data_news.loc[mask, "sub_category"] = \
            data_news.loc[mask, "sub_category"].replace(sc_map)

    return data_news


def remove_punct(tokens):
    """Remove punctuation from list of tokens"""
    return [token for token in tokens if re.match("\w", token)]


def remove_stopwords(tokens, stop_words):
    """Remove stop-words from list of tokens"""
    return [token for token in tokens if not token in stop_words]


def cleanup_contractions(tokens):
    """Remove contractions from list of tokens"""
    cleaned_tokens = []
    # Iterate over all tokens and keep track of the previous token
    prev_token = ""
    for token in tokens:
        if token in CONTRACTIONS.keys():
            # Handle cases depending on previous token
            if prev_token == "ca" and token == "n't":
                cleaned_tokens.pop()
                cleaned_tokens.append("cannot")
            elif prev_token == "wo" and token == "n't":
                cleaned_tokens.pop()
                cleaned_tokens.append("will")
                cleaned_tokens.append("not")
            # Use CONTRACTIONS map
            else:
                cleaned_tokens.append(CONTRACTIONS[token])
        else:
            cleaned_tokens.append(token)
        prev_token = token
    return cleaned_tokens


def concat_title_abstract(row):
    """Concatenate title and abstract of a datasample"""
    return f"{row['title']} {row['abstract']}"


def concat_all(row):
    """Concatenate title and abstract of a datasample"""
    return f"category: {row['category']}, sub-category: {row['sub_category']}. \
        title: {row['title']}, abstract: {row['abstract']}"


def get_survival_numbers(news_id, survival_data):
    """Get survival time and first-appearance timestamp

    Retrieve the data for news_id from the survival_data.

    Arguments:
        news_id -- news ID
        survival_data -- dict of dicts containing survival data for each 
        news ID

    Returns:
        the survival time in hours and the timestamp of the first
        appearance in an impression (closest to publish date).
    """
    # Handle nonexistent news id
    if news_id not in survival_data:
        return 0.0, None
    # Get survival numbers
    survival_numbers = survival_data[news_id]
    survival_time_hrs = survival_numbers["survival_time_hrs"]
    first_read_timestamp = survival_numbers["first_impression"]
    return survival_time_hrs, first_read_timestamp


def get_engagement_numbers(news_id, engagement_data):
    """Get number of times news were clicked/ignored/shown

    Retrieve the data for news_id from the engagement_data.

    Arguments:
        news_id -- news ID
        engagement_data -- dict of dicts containing engagement data for
        each news ID

    Returns:
        the number of times the article was clicked/ignored/shown
    """
    # Handle nonexistent news id
    if news_id not in engagement_data:
        return 0, 0, 0
    # Get engagement numbers
    engagement_numbers = engagement_data[news_id]
    clicked = engagement_numbers["clicked"]
    ignored = engagement_numbers["ignored"]
    shown = engagement_numbers["shown"]
    return clicked, ignored, shown


def preprocess_data_news(data_news, save_dir=None, exploration=False, behaviors_paths=None,
                         embedding=False):
    """Preprocessing routine for news data

    Arguments:
        data_news -- news data as pandas dataframe

    Keyword Arguments:
        save_dir -- where to save the data. If None, data 
        will not be saved (default: {None})
        exploration -- whether to construct columns for exploration 
        purposes (default: {False})
        behaviors_paths -- dict with paths to concatenated and trainfull
        behaviors data (default: {None})
        embedding -- whether to prepare data for embedding (default: {False})

    Raises:
        ValueError: missing behaviors_paths in exploration mode

    Returns:
        pandas dataframepre with preprocessed data
    """
    if exploration and behaviors_paths is None:
        raise ValueError(
            "behaviors_paths cannot be None when in exploration mode."
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

    # Replace NaN abstracts with ""
    print("[INFO] replacing NA abstracts with empty string")
    data_news["abstract"].fillna("", inplace=True)

    # Remap categories and sub-categories
    print("[INFO] remapping categories")
    data_news = _remap_categories(data_news)

    # Save to pickle, if save_dir provided
    if save_dir is not None:
        print("[INFO] saving")
        data_news.to_pickle(os.path.join(save_dir, "news.pkl"))

    # Prepare data for embedding, if wanted
    if embedding:
        print("[INFO] dropping columns irrelevant for embedding")
        # Drop unrequired columns, leave data_news unchanged
        data_news_emb = data_news.drop(
            columns=["url", "title_entities", "abstract_entities"],
            inplace=False
        )

        # Add column of concatenated title and abstract
        print("[INFO] concatenating title and abstract")
        data_news_emb["title_and_abstract"] = data_news_emb.progress_apply(
            concat_title_abstract,
            axis=1
        )

        # Add column of concatenated title and abstract
        print("[INFO] concatenating all")
        data_news_emb["all"] = data_news_emb.progress_apply(
            concat_all,
            axis=1
        )

        # Save to pickle, if save_dir provided
        if save_dir is not None:
            print("[INFO] saving data for embedding")
            # Store exploration data with 'emb_' prefix filename
            data_news_emb.to_pickle(os.path.join(save_dir, "emb_news.pkl"))

    # Collect additional exploratory data, if wanted
    if exploration:
        # Preprocess title and abstract text
        print("[INFO] preprocessing title and abstract")
        for col in ["title", "abstract"]:
            # Lowercase
            print(f"\t[INFO] lowercasing {col}")
            data_news[col] = data_news[col].str.lower()

            # Tokenize
            print(f"\t[INFO] tokenizing {col}")
            data_news[f"{col}_tokens"] = data_news[col].progress_apply(
                word_tokenize
            )

            # Remove punctuation
            print(f"\t[INFO] removing punctuation in {col}")
            data_news[f"{col}_tokens"] = data_news[f"{col}_tokens"].progress_apply(
                remove_punct
            )

            # Clean contractions
            print(f"\t[INFO] cleaning contractions in {col}")
            data_news[f"{col}_tokens"] = data_news[f"{col}_tokens"].progress_apply(
                cleanup_contractions
            )

            # Remove stopwords
            print(f"\t[INFO] removing stopwords in {col}")
            stop_words = set(stopwords.words("english"))
            data_news[f"{col}_tokens_no_stopwords"] = data_news[f"{col}_tokens"].progress_apply(
                remove_stopwords,
                args=[stop_words]
            )

        # Get length stats for title and abstract text
        print("[INFO] obtaining length stats for title and abstract")
        for col in ["title", "abstract"]:
            # Get length with/without stopwords
            data_news[f"{col}_length"] = data_news[f"{col}_tokens"].str.len()
            data_news[f"{col}_no_stopwords_length"] = data_news[f"{col}_tokens_no_stopwords"].str.len()

        # Sum title and abstract lengths
        data_news["title_and_abstract_length"] = data_news["title_length"] + \
            data_news["abstract_length"]
        data_news["title_and_abstract_no_stopwords_length"] = data_news["title_no_stopwords_length"] + \
            data_news["abstract_no_stopwords_length"]

        # Get corresponding behaviors dataframe
        print("[INFO] loading behaviors data")
        data_behaviors = read_pickled_data([
            behaviors_paths["survival"],
            "behaviors_concat.pkl"
        ])

        # Get survival data from behaviors data
        print("[INFO] obtaining survival data")
        survival_data = get_survival_data(data_behaviors)
        # Save survival data in news dataframe
        print("[INFO] processing survival data")
        survival_time_hrs, first_read_timestamp = zip(
            *data_news["news_id"].progress_apply(
                get_survival_numbers,
                args=[survival_data]
            )
        )
        data_news["survival_time_hrs"] = survival_time_hrs
        data_news["first_read_timestamp"] = first_read_timestamp

        # Get engagement data from behaviors data
        print("[INFO] loading behaviors data")
        data_behaviors = read_pickled_data([
            behaviors_paths["engagement"],
            "behaviors.pkl"
        ])
        # Save engagement data in news dataframe
        print("[INFO] obtaining engagement data")
        engagement_data = get_engagement_data(data_behaviors)
        print("[INFO] processing engagement data")
        clicked, ignored, shown = zip(
            *data_news["news_id"].progress_apply(
                get_engagement_numbers,
                args=[engagement_data]
            )
        )
        data_news["clicked"] = clicked
        data_news["ignored"] = ignored
        data_news["shown"] = shown
        # Compute engagement percentage
        data_news["engagement_percentage"] = data_news["clicked"] / \
            data_news["shown"] * 100
        # Handle division by 0
        data_news["engagement_percentage"].fillna(0.0, inplace=True)

        # Pickle data, if save_dir provided
        if save_dir is not None:
            print("[INFO] saving exploratory data")
            # Store exploration data with 'exp_' prefix filename
            data_news.to_pickle(os.path.join(save_dir, "exp_news.pkl"))

    return data_news


def get_unique_news_id_set(data_news):
    """Return the set of unique news IDs in the data"""
    return set(data_news["news_id"].unique())


def get_numeric_value_stats(data_news, columns, percentiles):
    """Collect numeric descriptors for given columns

    Arguments:
        data_news -- preprocessed news data as pandas dataframe
        columns -- which columns to collect stats for
        percentiles -- which percentiles to collect

    Returns:
        Stats in a pandas dataframe
    """
    stats = []
    for col in columns:
        stats.append(data_news[col].describe(percentiles=percentiles))
    stats_combined = pd.concat(stats, axis=1)
    return stats_combined
