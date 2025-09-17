# Working with DVC

This guide will help you initialize a DVC-enabled project, creating multiple dataset versions, and switching between them using `git` + `dvc`.
We will use the dataset `data/wine_quality_df.csv` as an example.


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

On Windows if you are using Powershell

```bash
.\dvc_practice\Scripts\Activate.ps1

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
git commit -m "feat(data): add wine_quality_df.csv to DVC tracking"
```

## Setting Up Remote Storage

Before setting up remote storage, ensure that the necessary dependencies are installed if you’re using a cloud provider, in addition to having programmatic access credentials and installing the library that allows you to connect to the cloud provider.

```bash
pip install 'dvc[gdrive]'
pip install 'dvc[s3]'
pip install 'dvc[gcs]'
pip install 'dvc[azure]'
```

## Setting up Remote Storage.

1. **Example of configuring Aws as remote storage:**

## Install the library

```bash
pip install awscli
```

## Enter configuration credentials (AWS Access Key Id, AWS Secret Access Key)

```bash
aws configure
```

**Note**
In this part your will be required to include your `AWS_ SECRET_ID` and `AWS_SECRET_KEY`, as well as defining the `region` you are working (by default `us-east-1` is selected) and the output of your preference (by default `json`).

```bash
dvc remote add -d s3_storage s3://{bucket_name}/{optional_folder}
```

2. **Example of configuring Google Drive as remote storage:**

```bash
dvc remote add --default gdrive-remote gdrive://<FOLDER-ID>
dvc remote modify gdrive-remote gdrive_client_id '<CLIENT-ID>'
dvc remote modify gdrive-remote gdrive_client_secret '<CLIENT-SECRET>'
```

**Note**
Currently, there are issues with authentication on Google’s side. And a couple of videos will be provided to explain how to get your `client_id` and `client_secret`:


3. **Example of configuring a local folder as remote storage:**

On another location in your computer, create a folder that will work as the remote storage:

```bash
mkdir -p ../local-dvc-storage
dvc remote add -d localstorage ../local-dvc-storage

git add .dvc/config
git commit -m "chore(dvc): configure remote storage (local)"
```

4. **Push Data to Remote**

``` bash
dvc push -r localstorage
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

### Modifying the File and Tracking Changes

For the dataset we are working with, we can remove records using a text editor, delete the header, or copy and paste more rows. Once the changes are made, we add them to DVC.

### Version 2 -- Trim to First 1000 Rows
```bash
# This will keep the header + 999 rows
{ head -n 1 data/wine_quality_df.csv; tail -n +2 data/wine_quality_df.csv | head -n 999; } > data/_tmp.csv
mv data/_tmp.csv data/wine_quality_df.csv

# Updating the tracking on Dvc
dvc add data/wine_quality_df.csv
git add data/wine_quality_df.csv.dvc
git commit -m "data: Version 2 of the dataset, with 999 rows with header"
dvc push -r gdrive-remote
```

---

### Version 3 -- Random 70% Sample

``` bash
python3 - <<'PYCODE'
import pandas as pd
df = pd.read_csv("data/wine_quality_df.csv")
df.sample(frac=0.7, random_state=42).to_csv("data/wine_quality_df.csv", index=False)
PYCODE

dvc add data/wine_quality_df.csv
git add data/wine_quality_df.csv.dvc
git commit -m "data: create v3 (random 70% sample)"
dvc push -r gdrive-remote
```

---

## Check History and Switch Versions

``` bash
git log --oneline -- data/wine_quality_df.csv.dvc

git checkout HEAD~2 -- data/wine_quality_df.csv.dvc && dvc checkout
git checkout HEAD~1 -- data/wine_quality_df.csv.dvc && dvc checkout
git checkout main -- data/wine_quality_df.csv.dvc && dvc checkout
```

------------------------------------------------------------------------

## 8. Versioning Workflow Diagram

``` text
       ┌─────────────┐
       │   Version 1 │  (Original dataset)
       └──────┬──────┘
              │
              ▼
       ┌─────────────┐
       │   Version 2 │  (Trimmed to 1000 rows)
       └──────┬──────┘
              │
              ▼
       ┌─────────────┐
       │   Version 3 │  (Random 70% sample)
       └─────────────┘

References:
HEAD      -> v3
HEAD~1    -> v2
HEAD~2    -> v1
```




## Best practices and additional recommendations
-   Every change made to data artifacts, when pushed to remote storage, will create a new folder where the metadata for those changes will be stored.
-   Always commit `.dvc` files before running `dvc push`.
-   Use descriptive commit messages (e.g. `data: add cleaned version with header removed`).
-   Avoid committing large data files directly to Git --- use DVC tracking.
