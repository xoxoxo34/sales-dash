import os
import sys
import subprocess

# List of required packages
packages = [
    "dash",
    "dash-bootstrap-components",
    "pandas",
    "numpy",
    "plotly",
    "reportlab"
]

# Function to install packages
def install_packages(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    install_packages(packages)