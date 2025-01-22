import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from collections import namedtuple

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

def normalize_signals(x):
    """
    Normalize the input signal data by its maximum power.
    """
    for i in range(x.shape[0]):
        max_power = np.sum(np.power(x[i, 0, :], 2) + np.power(x[i, 1, :], 2)) / x.shape[2]
        x[i] = x[i] / np.sqrt(max_power)
    return x

def load_dataset_from_file(file_path, num_device, num_samples_per_transmitter=1000):
    """
    Load and preprocess dataset from a pickle file for a specified number of devices.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    x_combined, y_combined = [], []
    for tx in range(num_device):
        tx_data = data['data'][tx]
        tx_data_swapped = np.transpose(tx_data, (0, 2, 1))  # Swap axes for correct format
        x_combined.append(tx_data_swapped)
        y_combined.extend([tx] * num_samples_per_transmitter)

    x = np.concatenate(x_combined, axis=0)
    x = normalize_signals(x)
    y = np.array(y_combined)
    return x, y

def get_dataset(category):
    """
    Load datasets for specified categories, either 'ORACLE' or 'WiSig'.
    Returns a structured dictionary or named tuple for organized access.
    """

    if category == 'ORACLE':
        dataset_dir = os.path.join(BASE_DIR, 'Dataset_ORALCE')
        run1_dir = os.path.join(dataset_dir, 'run1')
        run2_dir = os.path.join(dataset_dir, 'run2')

        # Training and validation data
        x_train, y_train = np.load(os.path.join(run1_dir, 'x_train_2ft.npy')), np.load(os.path.join(run1_dir, 'y_train_2ft.npy'))
        x_train, y_train = np.squeeze(x_train), np.squeeze(y_train)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=2023)

        # Test data for source
        x_test_s, y_test_s = np.load(os.path.join(run1_dir, 'x_test_2ft.npy')), np.load(os.path.join(run1_dir, 'y_test_2ft.npy'))
        x_test_s, y_test_s = np.squeeze(x_test_s), np.squeeze(y_test_s)

        # Test data for target
        x_test_t, y_test_t = np.load(os.path.join(run2_dir, 'x_test_2ft.npy')), np.load(os.path.join(run2_dir, 'y_test_2ft.npy'))
        x_test_t, y_test_t = np.squeeze(x_test_t), np.squeeze(y_test_t)

        return {
            'train': (x_train, y_train),
            'val': (x_val, y_val),
            'test_s': (x_test_s, y_test_s),
            'test_t': [(x_test_t, y_test_t)] 
        }

    elif category == 'WiSig':
        dataset_dir = os.path.join(BASE_DIR, 'Dataset_WiSig')

        # Training and validation data
        x, y = load_dataset_from_file(os.path.join(dataset_dir, 'rx_1-1_date1.pkl'), num_device=6)
        x, x_test_s, y, y_test_s = train_test_split(x, y, test_size=0.1, random_state=2023)
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=2023)

        # Test data from other dates
        x_test_t1, y_test_t1 = load_dataset_from_file(os.path.join(dataset_dir, 'rx_1-1_date2.pkl'), num_device=6)
        x_test_t2, y_test_t2 = load_dataset_from_file(os.path.join(dataset_dir, 'rx_1-1_date3.pkl'), num_device=6)
        x_test_t3, y_test_t3 = load_dataset_from_file(os.path.join(dataset_dir, 'rx_1-1_date4.pkl'), num_device=6)

        return {
            'train': (x_train, y_train),
            'val': (x_val, y_val),
            'test_s': (x_test_s, y_test_s),
            'test_t': [
                (x_test_t1, y_test_t1),
                (x_test_t2, y_test_t2),
                (x_test_t3, y_test_t3)
            ]
        }

    else:
        raise ValueError(f"Unsupported category: {category}")

if __name__ == "__main__":
    """
    Main block to test the get_dataset functionality.
    """
    for category in ['ORCALE', 'WiSig']:
        print(f"Testing {category} dataset...")
        dataset = get_dataset(category)

        # Train and validation data
        print(f"Train data shape: {dataset['train'][0].shape}, Labels: {len(dataset['train'][1])}")
        print(f"Validation data shape: {dataset['val'][0].shape}, Labels: {len(dataset['val'][1])}")

        # Test source data
        print(f"Source test data shape: {dataset['test_source'][0].shape}, Labels: {len(dataset['test_source'][1])}")

        # Test target data
        print("Target test data shapes:")
        for i, (x_test, y_test) in enumerate(dataset['test_target']):
            print(f"  Target test {i + 1} shape: {x_test.shape}, Labels: {len(y_test)}")