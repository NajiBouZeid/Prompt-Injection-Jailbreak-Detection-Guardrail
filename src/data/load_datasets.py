import os

from datasets import load_dataset
import pandas as pd


def load_hf_dataset(dataset_name: str) -> pd.DataFrame:
    """Load any HuggingFace dataset by name and return its first split as a pandas DataFrame."""
    dataset = load_dataset(dataset_name)
    split_name = list(dataset.keys())[0]
    print(f"{dataset_name}: available splits = {list(dataset.keys())}, using '{split_name}'")
    return dataset[split_name].to_pandas()


def save_raw_dataset(dataset_name: str, df: pd.DataFrame, raw_dir: str = "data/raw") -> str:
    """Save a dataset's DataFrame to disk exactly as-is (no normalization), as a parquet file."""
    os.makedirs(raw_dir, exist_ok=True)
    safe_name = dataset_name.replace("/", "__")
    path = os.path.join(raw_dir, f"{safe_name}.parquet")
    df.to_parquet(path, index=False)
    return path


def load_raw_dataset(dataset_name: str, raw_dir: str = "data/raw") -> pd.DataFrame:
    """Load a dataset previously saved by save_raw_dataset, from its local parquet file."""
    safe_name = dataset_name.replace("/", "__")
    path = os.path.join(raw_dir, f"{safe_name}.parquet")
    return pd.read_parquet(path)
