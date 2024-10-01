import pandas as pd
import os
import zipfile
import requests
from gin import configurable
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)


@configurable
class DataLoader:
    def __init__(self, data_url: str = None, local_csv_path: str = None, local_zip_path: str = "data.zip",
                 extract_dir: str = "extracted_data", extracted_file_name: str = None, delimiter: str = ',',
                 encoding: str = 'utf-8', error_bad_lines: bool = True, reload: bool = False):
        """
        Initializes the DataLoader with the given URL and optional parameters.

        :param data_url: URL of the zip file containing the data.
        :param local_csv_path: Path to an already existing local CSV file (if available).
        :param local_zip_path: Path to save the downloaded zip file (default is 'data.zip').
        :param extract_dir: Directory to extract the contents of the zip file (default is 'extracted_data').
        :param extracted_file_name: Desired name for the extracted CSV file (if specified).
        :param delimiter: Delimiter used in the CSV file (default is ',').
        :param encoding: Encoding format for reading the CSV file (default is 'utf-8').
        :param error_bad_lines: Whether to raise errors for bad lines (default is True).
        :param reload: Whether to download the data from URL if it already exists on disk (default is False).
        """
        self.data_url = data_url
        self.local_csv_path = local_csv_path
        self.local_zip_path = local_zip_path
        self.extract_dir = extract_dir
        self.extracted_file_name = extracted_file_name
        self.delimiter = delimiter
        self.encoding = encoding
        self.error_bad_lines = error_bad_lines
        self.reload = reload

        # Ensure the extract directory exists
        os.makedirs(self.extract_dir, exist_ok=True)

    def download_and_extract(self):
        """
        Downloads and extracts the zip file from the specified URL.
        """
        # If reload is False and the CSV file already exists, skip the download process
        if not self.reload and self.local_csv_path and os.path.exists(self.local_csv_path):
            logger.info(f"Reload is set to False and local CSV exists. Loading data from: {self.local_csv_path}")
            return self.local_csv_path

        # Check if the zip file already exists and reload is False, skip download
        if not self.reload and os.path.exists(self.local_zip_path):
            logger.info(
                f"Reload is set to False and zip file already exists. Using existing zip file: {self.local_zip_path}")
        else:
            logger.info(f"Downloading data from: {self.data_url}")
            response = requests.get(self.data_url)
            response.raise_for_status()  # Raise an error for bad responses

            # Write the content to a local zip file
            with open(self.local_zip_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Data downloaded and saved to: {self.local_zip_path}")

        # Extract the zip file to the specified directory
        with zipfile.ZipFile(self.local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(path=self.extract_dir)
            extracted_files = zip_ref.namelist()

            logger.info(f"Files in the archive: {extracted_files}")

            original_file_path = os.path.join(self.extract_dir, extracted_files[0])
            renamed_file_path = os.path.join(self.extract_dir, self.extracted_file_name)

            # Check if the renamed file already exists
            if os.path.exists(renamed_file_path):
                logger.info(f"The file '{self.extracted_file_name}' already exists. Skipping rename.")
            else:
                os.rename(original_file_path, renamed_file_path)
                logger.info(f"Renamed '{extracted_files[0]}' to '{self.extracted_file_name}'")

            return renamed_file_path

    def load(self):
        """
        Loads data from the CSV file, either from a local file, a zip archive, or downloads it from a URL.

        :return: DataFrame containing the loaded data.
        """
        try:
            # Use existing local CSV if reload is False and the file exists
            if not self.reload and self.local_csv_path and os.path.exists(self.local_csv_path):
                logger.info(f"Loading data from local CSV path: {self.local_csv_path}")
                return pd.read_csv(
                    self.local_csv_path,
                    delimiter=self.delimiter,
                    encoding=self.encoding,
                    on_bad_lines='error' if self.error_bad_lines else 'skip'
                )

            # If reload is True, or the CSV doesn't exist, download and extract
            extracted_file = self.download_and_extract()
            logger.info(f"Loading data from: {extracted_file}")
            data = pd.read_csv(
                extracted_file,
                delimiter=self.delimiter,
                encoding=self.encoding,
                on_bad_lines='error' if self.error_bad_lines else 'skip'
            )
            logger.info("Data loaded successfully")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {self.local_csv_path or extracted_file}")
            raise
        except pd.errors.ParserError:
            logger.error(f"Error parsing the file: {self.local_csv_path or extracted_file}")
            raise
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            raise
