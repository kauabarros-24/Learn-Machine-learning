from pathlib import Path
import pandas as pd

def load_housing_data():
    base_dir = Path(__file__).resolve().parent
    tarball_path = base_dir / "datasets/housing.csv"

    return pd.read_csv(tarball_path)

housing = load_housing_data()
housing.info()