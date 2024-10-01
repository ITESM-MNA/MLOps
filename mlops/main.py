import argparse
import gin
import logging
from ydata_profiling import ProfileReport
from mlops.dataset.data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


def generate_profile_report(df, report_title, reports_dir, report_filename="amphibians_profile_report.html"):
    """
    Generates and saves a profiling report using ydata_profiling.

    :param df: DataFrame to profile
    :param report_title: Title of the profiling report
    :param reports_dir: Directory to save the report
    :param report_filename: The name of the output report file
    """
    logger.info("Generating profiling report...")
    profile = ProfileReport(df, title=report_title, explorative=True)

    # Ensure the reports_dir is absolute and exists
    reports_dir = reports_dir.resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save the profiling report
    report_path = reports_dir / report_filename
    profile.to_file(report_path)
    logger.info(f"Profile report generated and saved to: {report_path}")


def run():
    """
    Main function to run the data loading and profiling report generation process.
    """
    # Initialize the DataLoader
    data_loader = DataLoader()

    # Load the dataset
    df = data_loader.load()

    # Check if report generation is enabled from gin config
    if gin.query_parameter('%generate_report'):
        # Resolve the reports directory path
        reports_dir = (data_loader.base_dir / gin.query_parameter('reports_dir')).resolve()
        # Generate and save the profiling report
        generate_profile_report(df, "Amphibians Data Profiling Report", reports_dir)
    else:
        # Log that report generation is disabled
        logger.info("Report generation is disabled by the gin configuration.")


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add a command-line argument for the configuration file path
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Parse the gin configuration file specified by the command-line argument
    gin.parse_config_file(args.config)

    # Run the profiling task
    run()
