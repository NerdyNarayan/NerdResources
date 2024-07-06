# Tip for running the notebooks in Vscode

**Step 1: Install Python**

If you haven't installed Python on your system, download and install it from the official Python website: <https://www.python.org/downloads/>

**Step 2: Create a Virtual Environment**

Open a terminal or command prompt and run the following command to create a new virtual environment:

```
python -m venv myenv
```

Replace `myenv` with the name of your virtual environment.

**Step 3: Activate the Virtual Environment**

To activate the virtual environment, run the following command:

```
myenv\Scripts\activate  (on Windows)
source myenv/bin/activate  (on macOS/Linux)
```

You should see a notification indicating that you are now using the virtual environment.

**Step 4: Install pip**

pip is the package installer for Python. If you don't have pip installed, run the following command:

```
python -m ensurepip
    or
apt-get install python-pip (on Ubuntu/Debian)
pacman -S python-pip (on Arch Linux)

```

**Step 5: Install necessary packages**

Run the following command to install the necessary packages for machine learning:

```
pip install numpy scipy scikit-learn pandas matplotlib seaborn tensorflow keras
```

This will install the following packages:

- NumPy: a library for efficient numerical computation
- SciPy: a library for scientific computing and signal processing
- scikit-learn: a machine learning library
- Pandas: a library for data manipulation and analysis
- Matplotlib: a library for data visualization
- Seaborn: a visualization library built on top of Matplotlib
- TensorFlow: a deep learning library
- Keras: a deep learning library

**Step 6: Verify the installation**

To verify that the packages have been installed correctly, run the following commands:

```
python -c "import numpy; print(numpy.__version__)"
python -c "import pandas; print(pandas.__version__)"
python -c "import tensorflow; print(tensorflow.__version__)"
```

This will print the version numbers of the installed packages.

**Step 7: Deactivate the Virtual Environment**

When you're finished using the virtual environment, deactivate it by running the following command:

```
deactivate
```

This will return you to your system's default Python environment.

**Step 8: Open the Notebook in VSCode**
Open any of the folder or notebook in vscode. In vscode install the Jupyter extension pack to have all the tools and utilities required for running a jupyter notebook efficiently. After install
