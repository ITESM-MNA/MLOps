
# Initial Setup for the MLOps Course of the Master's in Applied Artificial Intelligence

This document provides step-by-step instructions to get started with the repository as it will be used in the course. We'll begin with how to clone the GitHub repository, install Python 3.13, and create a Python virtual environment.

## Prerequisites

- **Git** must be installed. You can download it from [git-scm.com](https://git-scm.com/).
- **Python 3.13** must be installed. You can download it from [python.org](https://www.python.org/downloads/).

## Clone the Repository

1. **Open a terminal (Command Prompt, PowerShell, macOS Terminal, or a terminal in Linux).**
2. **Navigate to the directory where you want to clone the repository:**

   ```bash
   cd path/to/directory/
   ```

3. **Clone the repository:**

   ```bash
   git clone https://github.com/ITESM-MNA/MLOps
   ```

   This will create a folder with the repository's name in the specified directory.

4. **Navigate to the cloned repository directory (MLOps):**

   ```bash
   cd repository_name
   ```

   Replace `repository_name` with the actual name of the cloned repository.

## Setting up the Virtual Environment with Python 3.13

### Verify the Installation of Python 3.13

Before creating the virtual environment, make sure that Python 3.13 is installed and accessible from your terminal:

```bash
python3.13 --version
```

You should see something like `Python 3.13.x`. If it's not installed, make sure to download and install Python 3.13 from [python.org](https://www.python.org/downloads/).


> **Note on Python and pip commands:**  
> Depending on your operating system and how Python is installed, you may see commands like `python` or `python3` (and similarly `pip` or `pip3`).  
>
> - On many **Linux/macOS systems**, `python` may still point to Python 2.x, so we use `python3` to explicitly call Python 3.  
> - On **Windows**, the command `python` usually points directly to the latest installed Python 3 version, but sometimes you can also use the launcher with `py -3.11`.  
> - For package management, `pip3` ensures you are installing packages for Python 3, while `pip` may default to Python 2 in some environments.  
>
> In short:  
> - Use `python3` and `pip3` when working on macOS/Linux (to be explicit).  
> - Use `python` or `py -3.11` and `pip` on Windows, unless otherwise specified.



### Create and Activate the Virtual Environment

1. **Install virtualenv with pip**
   
   First, we need to install virtualenv library, this could be done with the following commands executed in your terminal:

   ```bash
   pip install virtualenv
   ```

   or

   ```bash
   pip3 install virtualenv 
   ```

2. **Create the Virtual Environment using Python 3.13:**

   In the terminal, run:

   ```bash
   python3.13 -m venv virtual_environment_name
   ```

   Replace `nombre_del_entorno_virtual` with your preferred name for your virtual environment (e.g. `itesm_venv`).

3. **Activate the Virtual Environment:**

   - **On Windows:**

     ```bash
     .\virtual_environment_name\Scripts\activate
     ```

     or if using Git Bash

     ```bash
     source \virtual_environment_name\Scripts\activate
     ```

   - **On Linux and MacOS:**

     ```bash
     source virtual_environment_name/bin/activate
     ```

   After activating the environment, you should see the environment's name in parentheses before the path in your terminal, indicating that the virtual environment is active.

4. **Deactivate the Virtual Environment:**

   To exit the virtual environment, simply run:

   ```bash
   deactivate
   ```

## Installing Dependencies

Once the virtual environment is active, you can install the necessary dependencies for the project using the `requirements.txt` file that contains the listed dependencies. You can install them with:

```bash
pip install -r requirements.txt
```

## Additional Notes

- Make sure to activate the virtual environment every time you work on your project to use the correct versions of the libraries.
- Keep your dependencies updated and regularly check the `requirements.txt` file.

Now you're ready to start working on the course with the virtual environment properly set up!
