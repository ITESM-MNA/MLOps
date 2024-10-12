# Setting up ML project structure

This guide will help you set up a project structure for working with Machine Learning projects, also, you will have an understanding about different options for refactoring your code which will make you available to execute your first ML pipelins from the command line.

If you haven´t done it, install from pip the following packages:

```bash
pip install cookiecutter cookiecutter-data-science
```

- Cookiecutter: This library will help you with templates for different types of projects, not only data oriented, but software projects as well, you only need to look for the template that better works with your requirements.

- Cookiecutter-data-science: This library is an upgrade for the previous template available for data science projects.



1. **Defining the template structure you want to use**

You can use different templates available for structuring your project with Cookiecutter, for this activity, we will use the setup of Version 2 from the Cookiecutter Data Science template, after typing it in the terminal, it will show different questions about:
- Project name.
- Repository name.
- Module name.
- Author name.
- Description of the project.
- Python version number.
- Dataset storage.
- Environment manager.
- Dependency file.
- Open source license.

From the terminal in your parent directory, you need to type

```bash
ccds
```

The next links provide other templates; just be aware that those links use version 1 of the Cookiecutter Data Science template.


- **MLOps template** You can download the structure with:

```bash
cookiecutter https://github.com/mlops-guide/mlops-template.git
```

- **Khuyen Tran template** You can download the structure with:

```bash
cookiecutter https://github.com/khuyentran1401/data-science-template
```


## Cookiecutter Data Science version 2 project structure

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         refactoring and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── refactoring   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes refactoring a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```



## Setting Up the working Envnvironment

2. **Get into the project folder you named in the previous step**

```bash
cd path/to/directory/
```

3. **Create a virtual environment:**

```bash
python -m venv practice1
```

4. **Activate the virtual environment:**

On Mac:
```bash
source practice1/bin/activate
```

On Windows:
```bash
practice1\Scripts\activate
```

If you want to make your virtual environment available as a kernel for using in your Jupyter notebooks, execute the following code in the terminal.

```bash
python -m ipykernel install --user --name=practice1 --display-name="Python (practice1)"
```

Note:
In the display name, you could use any descriptive name you want, the value of the **--name** argument must refers to the name of your actual virtual environment.

You can verify that the instalation of your environment is done correctly executing the following code:

```
jupyter kernelspec list
```


5. **Initialize a git repository:**

```bash
git init
```

6. **Upgrade pip (Optional)**

```bash
python -m pip install --upgrade pip
```

7. **Install necessary packages:**

```bash
 pip install numpy pandas ipykernel matplotlib seaborn scikit-learn notebook dvc mlflow ucimlrepo
```


If you have not download the data, follow the instructions on the notebook **1.2.1.0 Getting the data.ipynb**.

Once you get the data you will be working on, you can refer to the **Data Versioning.md** file to initialize your Dvc repository, add your remote storage (if you are using some cloud provider) or using a local storage and add your dataset to Dvc tracking.


# Wine Quality ML development

For this session, we will start with the notebook **Wine_EDA.ipynb**, which explains what would be the normal thinking process when developing ML projects.

Then, you will see two more notebooks in this session:
- Wine_Rafactored_V1: This notebook includes the ML pipeline developed using functions.
- Wine_Rafactored_V2: This notebook includes the ML pipeline developed using classes.

Be aware that both notebooks are there only for the purpose of the class, so you can see in an interactive way, how they work and both will be a starting point for the main objective of the session.

The code available in both notebooks, is replicated with minor adjustments in two python scripts of the same name.

After looking into the details of the previous notebooks, we are ready to start why modularization and reproducibility (which is a topic that we will cover in deep in later sessions). For this, you will find four python scripts:
- load_data.py
- preprocess_data.py
- evaluate.py
- train.py

And two yaml files:
- params.yaml
- dvc.yaml

In the command line of a Dvc repo, if you execute:

```bash
dvc dag
```

You will see the steps that are included in your pipeline and where defined in the dvc.yaml file. And finally for this session, if you execute

```bash
dvc repro
```

The whole pipeline will be executed following both yaml files instructions.

Note:
If the pipeline does not run as expected, consider that some of the next scenarios may be happening:
- The working directory is not running on a dvc repo.
- How Windows, Mac o Linux deal with directories.
- The cookiecutter template you choose at the end, may be different to the one I choose for this activity, that make the paths different.


# Integrating DVC with MLflow

On the same folder, you will find new versions of the refactored files we saw in the last class; those files still focused on the refactoring of the Wine_Refactored_V1.ipynb with the usage of functions, but are adjusted to integrate MLflow with Dvc, so you can track your experiments on the MLflow UI, everytime yoyuurun **dvc_repro**.

As well, you can find a new version of the params.yaml (params_v2.yaml) and the dvc.yaml (dvc_v2.yaml) that includes the adjustments as well for running not only the Logistic Regression, but a Random Forest model as well, so in the first run of the pipeline, both models will run, and in the next iterations, only if you change some of the arguments of one model in the parameters_v2.yaml file, that will be the only model that will run.

Consider that if you want to run this new versions, those may be running on a different folder, with a similar structure to the one you have already defined, so Dvc won't get confused between params.yaml and params_v2.yaml (same for the dvc.yaml files).

Remember that *mlflow* must be installed and once you have installed, it should be initialized from a terminal.

```bash
pip install mlflow
```

Then, for initializing *mlflow*

```bash
mlflow ui
```

After this, you now would be able to run mlflow on the IP **http://127.0.0.1:5000**, then, you are ready to run your pipeline with **dvc repro**.

--------

