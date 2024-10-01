import argparse
import gin
import logging
from pathlib import Path
from ydata_profiling import ProfileReport
from mlops.dataset.data_loader import DataLoader  # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Register reports_dir as a constant
gin.constant('reports_dir', 'reports')


def run():
    # Initialize the DataLoader
    data_loader = DataLoader()

    # Load the dataset
    df = data_loader.load()

    # Generate a profile report
    logger.info("Generating profiling report...")
    profile = ProfileReport(df, title="Amphibians Data Profiling Report", explorative=True)

    # Get reports directory from gin configuration
    reports_dir = Path(gin.query_parameter('reports_dir'))

    # Ensure the reports_dir is absolute by resolving against base_dir
    reports_dir = (data_loader.base_dir / reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save the profiling report
    report_path = reports_dir / "amphibians_profile_report.html"
    profile.to_file(report_path)
    logger.info(f"Profile report generated and saved to: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the config file')  # Updated argument
    args = parser.parse_args()

    # Parse the gin configuration file
    gin.parse_config_file(args.config)

    # Run the profiling task
    run()
