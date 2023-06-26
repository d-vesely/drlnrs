from datasets import Dataset
import numpy as np
import openai
import os
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
tqdm.pandas()


def get_huggingface_dataset(df_data, device):
    """Turn pandas dataframe to huggingface dataset"""
    hf_data = Dataset.from_pandas(df_data)
    # Use torch and send data to device
    hf_data = hf_data.with_format("torch", device=device)
    return hf_data


def get_dataloader(hf_data, batch_size):
    """Create dataloader for given huggingface dataset"""
    #! Do not shuffle, it would interfere with construction of embeddings_map
    #! in embed_news
    return DataLoader(hf_data, batch_size=batch_size, shuffle=False)


def mean_pooling(model_output, attention_mask):
    """Perform mean pooling
    Code copied from: https://www.sbert.net/examples/applications/computing-embeddings/README.html
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1) \
        .expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def embed_news(data_news, checkpoint, save_dir=None, to_embed="title",
               max_length=256, batch_size=128):
    """Create embeddings for each news

    Arguments:
        data_news -- news data as pandas dataframe
        checkpoint -- sentence transformer model checkpoint. List of pretrained
        models available here: https://www.sbert.net/docs/pretrained_models.html

    Keyword Arguments:
        save_dir -- where to save the data. If None, data 
        will not be saved (default: {None})
        to_embed -- which part of news data to embed (default: {"title"})
        max_length -- maximum number of tokens per text (default: {256})
        batch_size -- batch size for embedding (default: {32})

    Raises:
        ValueError: to_embed value is not one of:
        ["title", "abstract", "title_and_abstract"]

    Returns:
        dict mapping news IDs to their embedding
    """
    if to_embed not in ["title", "abstract", "title_and_abstract", "all"]:
        raise ValueError(f"to_embed must be one of: {to_embed}")

    # Construct save_dir, if provided
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "embeddings")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"[INFO] embeddings will be saved in {save_dir}")
    else:
        print("[WARN] embeddings will not be saved")

    # Get torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # Prepare pretrained model and tokenizer
    print(f"[INFO] loading model: {checkpoint}")
    model = AutoModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model.to(device)

    # Get huggingface dataset from pandas dataframe
    print(f"[INFO] preparing dataloader")
    hf_data_news = get_huggingface_dataset(data_news, device)
    dataloader = get_dataloader(hf_data_news, batch_size=batch_size)

    # Incrementally fill emebeddings tensor
    embeddings = torch.empty((0, 768)).to(device)

    print(f"[INFO] embedding: {to_embed}")
    for batch in tqdm(dataloader):
        # Get list of texts from batch
        texts = batch[to_embed]

        # Tokenize the texts, return pytorch tensors
        encoded_input = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        encoded_input.to(device)

        # Feed tokens into model
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Mean pool the output
        batch_embeddings = mean_pooling(
            model_output,
            encoded_input["attention_mask"]
        )

        # Append batch of embeddings
        embeddings = torch.cat((embeddings, batch_embeddings), dim=0)

    # Construct dict, mapping news IDs to their embedding
    # This relies on the fact that the batches were sequential and not shuffled
    print("[INFO] creating embeddings map")
    embeddings = embeddings.cpu()
    embeddings_map = {}
    for i, news_id in tqdm(enumerate(hf_data_news["news_id"])):
        embeddings_map[news_id] = embeddings[i]

    # Save with torch, if save_dir provided
    print("[INFO] saving")
    if save_dir is not None:
        torch.save(
            embeddings_map,
            os.path.join(save_dir, f"{to_embed}_emb_map.pt")
        )

    return embeddings_map


def embed_news_openai(data_news, save_dir=None):
    model = "text-embedding-ada-002"
    embeddings_map = {}

    i = 0
    texts = []
    ids = []
    for row in tqdm(data_news.itertuples(), total=len(data_news)):
        news_id = row.news_id
        title_and_abstract = row.title_and_abstract
        title_and_abstract = title_and_abstract.replace("\n", " ")
        texts.append(title_and_abstract)
        ids.append(news_id)
        i += 1
        if i % 1000 == 0:
            embedding = openai.Embedding.create(
                input=texts,
                model=model
            )
            for j in range(1000):
                embeddings_map[ids[j]] = torch.tensor(
                    embedding["data"][j]["embedding"])
            ids = []
            texts = []

    if ids != []:
        embedding = openai.Embedding.create(
            input=texts,
            model=model
        )
        for j in range(len(ids)):
            embeddings_map[ids[j]] = torch.tensor(
                embedding["data"][j]["embedding"])

    print("[INFO] saving")
    if save_dir is not None:
        torch.save(
            embeddings_map,
            os.path.join(save_dir, "embeddings",
                         "title_and_abstract_openai_emb_map.pt")
        )

    return embeddings_map


def embed_categories(data_news, checkpoint, save_dir=None, batch_size=128):
    # Construct save_dir, if provided
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "embeddings")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"[INFO] embeddings will be saved in {save_dir}")
    else:
        print("[WARN] embeddings will not be saved")

    # Get torch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # Prepare pretrained model and tokenizer
    print(f"[INFO] loading model: {checkpoint}")
    model = AutoModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model.to(device)

    # Get huggingface dataset from pandas dataframe
    print(f"[INFO] preparing dataloader")
    hf_data_news = get_huggingface_dataset(data_news, device)
    dataloader = get_dataloader(hf_data_news, batch_size=batch_size)

    returns = []
    for to_embed in ["category", "sub_category"]:
        # Incrementally fill emebeddings tensor
        embeddings = torch.empty((0, 384)).to(device)

        print(f"[INFO] embedding: {to_embed}")
        for batch in tqdm(dataloader):
            # Get list of texts from batch
            texts = batch[to_embed]

            # Tokenize the texts, return pytorch tensors
            encoded_input = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            encoded_input.to(device)

            # Feed tokens into model
            with torch.no_grad():
                model_output = model(**encoded_input)

            # Mean pool the output
            batch_embeddings = mean_pooling(
                model_output,
                encoded_input["attention_mask"]
            )

            # Append batch of embeddings
            embeddings = torch.cat((embeddings, batch_embeddings), dim=0)

        # Construct dict, mapping news IDs to their embedding
        # This relies on the fact that the batches were sequential and not shuffled
        print("[INFO] creating embeddings map")
        embeddings = embeddings.cpu()
        embeddings_map = {}
        for i, news_id in tqdm(enumerate(hf_data_news["news_id"])):
            embeddings_map[news_id] = embeddings[i]

        # Save with torch, if save_dir provided
        print("[INFO] saving")
        if save_dir is not None:
            torch.save(
                embeddings_map,
                os.path.join(save_dir, f"{to_embed}_emb_map.pt")
            )

        returns.append(embeddings_map)

    return returns


def one_hot_encode_categories(data_news, save_dir=None):
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "embeddings")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"[INFO] embeddings will be saved in {save_dir}")
    else:
        print("[WARN] embeddings will not be saved")

    category_index_map = {
        "lifestyle": 0,
        "news": 1,
        "health": 2,
        "sports": 3,
        "weather": 4,
        "entertainment": 5,
        "food and drink": 6,
        "autos": 7,
        "travel": 8,
        "video": 9,
        "finance": 10,
        "tv": 11,
        "movies": 12,
        "music": 13,
        "kids": 14,
    }

    embeddings_map = {}

    for row in tqdm(data_news.itertuples(), total=len(data_news)):
        news_id = row.news_id
        category = row.category
        embedding = torch.zeros(15)
        category_index = category_index_map[category]
        embedding[category_index] = 1.0
        embeddings_map[news_id] = embedding

    print("[INFO] saving")
    if save_dir is not None:
        torch.save(
            embeddings_map,
            os.path.join(save_dir, "category_1hot_map.pt")
        )

    return embeddings_map


def load_embeddings(save_dir, to_embed="title"):
    """Load embeddings map saved with torch"""
    embeddings_map = torch.load(
        os.path.join(
            save_dir,
            "embeddings",
            f"{to_embed}_emb_map.pt"
        )
    )
    return embeddings_map


def build_feature_vectors(data_news, feature_columns, map_name, save_dir=None):
    # Construct save_dir, if provided
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "embeddings")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"[INFO] embeddings will be saved in {save_dir}")
    else:
        print("[WARN] embeddings will not be saved")

    print("[INFO] converting timestamp column")
    data_news["first_read_timestamp"] = data_news["first_read_timestamp"].view(
        np.int64)
    data_news["first_read_timestamp"] = data_news["first_read_timestamp"].progress_apply(
        lambda x: max(0, x / 3.6e12)
    )

    features_map = {}

    print("[INFO] building features map")
    for _, row in tqdm(data_news.iterrows(), total=data_news.shape[0]):
        news_id = row["news_id"]
        feature_vector = torch.zeros(len(feature_columns))
        for i, c in enumerate(feature_columns):
            feature_vector[i] = row[c]
        features_map[news_id] = feature_vector

    print("[INFO] saving")
    if save_dir is not None:
        torch.save(
            features_map,
            os.path.join(save_dir, f"{map_name}_features_map.pt")
        )

    return features_map


def load_feature_vectors(save_dir, map_name):
    """Load embeddings map saved with torch"""
    features_map = torch.load(
        os.path.join(
            save_dir,
            "embeddings",
            f"{map_name}_features_map.pt"
        )
    )
    return features_map
