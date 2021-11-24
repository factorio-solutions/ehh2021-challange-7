import glob
from pathlib import Path

import pandas as pd


def __load_ikem_data(data_folder):
    all_files = glob.glob(str(data_folder / 'ikem' / "*.xlsx"))

    for filename in all_files:
        df: pd.DataFrame = pd.read_excel(filename)
        df['rodné číslo'] = pd.util.hash_pandas_object(df['rodné číslo'])
        df.to_csv(filename, index=False)


if __name__ == '__main__':
    data_folder_ = 'mnt'
    __load_ikem_data(Path(data_folder_))
