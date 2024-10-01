import argparse
import gin
import logging
from mlops.data.data_loader import DataLoader
from IPython.display import display
from IPython.core.display import Markdown

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])


def run():
    # Initialize the DataLoader instance using the gin configuration
    data_loader = DataLoader()

    # Load the data using the DataLoader
    df = data_loader.load()

    # Display the first few rows as a confirmation
    logging.info("Data loaded successfully. Here are the first few rows:")
    display(df.head())  # Displaying the DataFrame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    # Parse the gin configuration file
    gin.parse_config_file(args.config)

    # Run the data loading process
    run()
