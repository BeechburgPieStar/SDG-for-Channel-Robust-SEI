import os
import numpy as np
from sklearn.model_selection import train_test_split

BASE_PATH = os.path.join(os.path.dirname(__file__), "..")
DATASET_PATH = os.path.join(BASE_PATH, "dataset")

def load_dataset(file_path):
    """
    Load and squeeze a numpy dataset.

    Parameters:
    - file_path: Path to the dataset file.

    Returns:
    - Squeezed numpy array.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return np.squeeze(np.load(file_path))

def get_dataset_paths(run, data_type, ft):
    """
    Construct the file paths for the given run and feature type.

    Parameters:
    - run: Run number (e.g., 1).
    - data_type: 'train' or 'test'.
    - ft: Feature type (e.g., 2ft).

    Returns:
    - Tuple of paths to the x and y dataset files.
    """
    x_path = os.path.join(DATASET_PATH, f"run{run}", f"x_{data_type}_{ft}.npy")
    y_path = os.path.join(DATASET_PATH, f"run{run}", f"y_{data_type}_{ft}.npy")
    return x_path, y_path

def TrainDataset(mark, test_size=0.3, shuffle=True, random_state=2023):
    """
    Load and split the training dataset into train and validation sets.

    Parameters:
    - mark: List of [run, ft] to generate dataset paths.
    - test_size: Proportion of data for validation.
    - shuffle: Whether to shuffle the data before splitting.
    - random_state: Seed for reproducibility.

    Returns:
    - x_train, x_val, y_train, y_val: Split training and validation sets.
    """
    try:
        x_path, y_path = get_dataset_paths(mark[0], "train", f"{mark[1]}ft")
        x = load_dataset(x_path)
        y = load_dataset(y_path)

        return train_test_split(
            x, y, test_size=test_size, shuffle=shuffle, random_state=random_state
        )
    except Exception as e:
        print(f"Error loading training dataset: {e}")
        raise

def TestDataset(mark):
    """
    Load the test dataset.

    Parameters:
    - mark: List of [run, ft] to generate dataset paths.

    Returns:
    - x_test, y_test: Test data and labels.
    """
    try:
        x_path, y_path = get_dataset_paths(mark[0], "test", f"{mark[1]}ft")
        x = load_dataset(x_path)
        y = load_dataset(y_path)

        return x, y
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        raise