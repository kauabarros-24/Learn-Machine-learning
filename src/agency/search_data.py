from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]



def load_housing_data():
    base_dir = Path(__file__).resolve().parent
    tarball_path = base_dir / "datasets/housing.csv"

    return pd.read_csv(tarball_path)

housing = load_housing_data()
housing.hist(bins=50, figsize=(12, 8))
plt.show()

housing.info()

train_set, test_set = shuffle_and_split_data(housing, 0.2)
print(train_set, test_set)