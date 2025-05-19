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
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully.\n")
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}. Please check your internet connection or pip setup.\n")

if __name__ == "__main__":
    install_packages(packages)
