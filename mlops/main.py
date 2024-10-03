import argparse
import gin
import logging
from ydata_profiling import ProfileReport
from mlops.dataset.data_loader import DataLoader
from mlops.feature_engineering.data_preprocessing import DataPreprocessing  # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def generate_profile_report(df, report_title, reports_dir, report_filename="amphibians_profile_report.html"):
    """
    Generates and saves a profiling report using ydata_profiling.
    """
    logger.info("Generating profiling report...")
    profile = ProfileReport(df, title=report_title, explorative=True)

    reports_dir = reports_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / report_filename
    profile.to_file(report_path)
    logger.info(f"Profile report generated and saved to: {report_path}")


def run():
    """
    Main function to run the data loading, preprocessing, and profiling report generation process.
    """
    # Initialize the DataLoader
    data_loader = DataLoader()

    # Load the dataset
    df = data_loader.load()

    # Preprocess the data
    data_preprocessor = DataPreprocessing(df)
    X_pca, y = data_preprocessor.preprocess()

    # Check if report generation is enabled from gin config
    if gin.query_parameter('%generate_report'):
        reports_dir = (data_loader.base_dir / gin.query_parameter('reports_dir')).resolve()
        generate_profile_report(df, "Amphibians Data Profiling Report", reports_dir)
    else:
        logger.info("Report generation is disabled by the gin configuration.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')
    args = parser.parse_args()

    gin.parse_config_file(args.config)
    run()
