# Working with DVC

This guide will help you initialize a DVC environment and gain hands-on experience.

## Prerequisites and Recommendations

To work with DVC, it must be initialized in a Git repository. Therefore, it’s important to have Git installed, and if you haven’t cloned the course repository yet, you should have a folder where Git is initialized beforehand. Additionally, the following example builds on the notebook 1.2.1.0 Gettting_the_data.ipynb, where the wine dataset is downloaded from the [UCI Irvine Machine Learning Repository](https://archive.ics.uci.edu/).

Except for reviewing the notebook 1.2.1.0 Gettting_the_data.ipynb, the rest of the activity will be carried out from the command line interface (CLI).

If you haven’t followed the instructions in section 1.1_Introduction, consider the following steps:

## Setting Up the Project to Work with Virtual Environments, Git, and DVC

1. **Create a virtual environment:**

```bash
python -m venv dvc_practice
```

2. **Activate the virtual environment:**

On Mac/Linux:
```bash
source dvc_practice/bin/dvc_activate
```

On Windows:
```bash
.\dvc_practice\Scripts\activate
```

If using Git Bash on Windows, the following command also works:

```bash
source \dvc_practice\Scripts\activate
```


3. **Initialize the Git repository:**

```bash
git init
```

4. **Upgrade pip:**

```bash
pip install --upgrade pip
```

5. **Install the minimum required libraries:**
If you haven't installed the `requirements.txt` file, the following libraries would be the minimum necessary to work with.

```bash
pip install numpy pandas ipykernel matplotlib seaborn dvc
```

## Initializing DVC and Including Data

1. **Initialize DVC:**

```bash
dvc init
```

Initializing DVC will create several files where metadata for data artifacts (data, models, metrics) will be stored.

2. **Add the wine dataset to DVC tracking:**

```bash
dvc add wine_quality_df.csv
```

3. **Stage the changes in Git:**

```bash
git add wine_quality_df.csv.dvc .gitignore
```

4. **Commit the changes:**

```bash
git commit -m "Adding raw data"
```

## Setting Up Remote Storage

Before setting up remote storage, ensure that the necessary dependencies are installed if you’re using a cloud provider, in addition to having programmatic access credentials and installing the library that allows you to connect to the cloud provider.

```bash
pip install 'dvc[gdrive]'
pip install 'dvc[s3]'
pip install 'dvc[gcs]'
pip install 'dvc[azure]'
```

### Example with Aws

## Install the library

```bash
pip install awscli
```

## Enter configuration credentials (AWS Access Key Id, AWS Secret Access Key)

```bash
aws configure
```


1. **Example of configuring Google Drive as remote storage:**

```bash
dvc remote add -d remote-storage gdrive://{Google Drive folder ID}
```

**Note**
Currently, there are issues with authentication on Google’s side. If authentication with Google Drive fails, you may consider the following options:


1. **Example of configuring a local folder as remote storage:**

```bash
dvc remote add -d local_remote /Users/your_username/Documents/Demo1/local_storage
```

1. **Example of an Amazon S3 bucket as remote storage:**

```bash
dvc remote add -d s3_storage s3://{bucket_name}/{optional_folder}
```

2. **Commit the remote storage configuration:**

```bash
git add .dvc/config
git commit -m "Setting up remote"
```

3. **Upload data to remote storage:**

```bash
dvc push
```

## Fetching Data from Remote Storage

We will simulate fetching data from local storage.

1. **You can delete the cache and the dataset:**

```bash
rm -rf wine_quality_df.csv
rm -rf .dvc/cache
```

2. **Retrieve the data from remote storage:**

```bash
dvc pull
```

## Modifying the File and Tracking Changes

For the dataset we are working with, we can remove records using a text editor, delete the header, or copy and paste more rows. Once the changes are made, we add them to DVC.

1. **Add changes to DVC:**

```bash
dvc add wine_quality_df.csv
```

2. **Stage the changes in Git:**

```bash
git add wine_quality_df.csv.dvc
```

3. **Commit the changes:**

```bash
git commit -m "Removing lines"
```

4. **Push the changes to remote storage:**

```bash
dvc push
```

**Note**
Every change made to data artifacts, when pushed to remote storage, will create a new folder where the metadata for those changes will be stored.

## Retrieve a Previous Version of the Dataset

To work with a previous version of the dataset and fetch it from remote storage:

1. **Checkout the previous version of the `.dvc` file:**

```bash
git checkout HEAD^1 wine_quality_df.csv.dvc
```

2. **Update the local data to match the version we just checked out:**

```bash
dvc checkout
```

3. **Commit the changes:**

```bash
git commit -m "Reverting changes"
```