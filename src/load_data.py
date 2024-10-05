import pandas as pd
import sys

def load_data(filepath):
    return pd.read_csv(filepath)

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_file = sys.argv[2]
    data = load_data(data_path)
    data.to_csv(output_file, index=False)
