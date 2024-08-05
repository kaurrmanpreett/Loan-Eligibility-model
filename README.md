# Loan-Eligibility-model-solution
This repository contains a solution for predicting loan eligibility based on various factors. The project is structured to provide a clear workflow from data loading and preprocessing to model training and evaluation.


## Modules

- **data_loader.py**: Contains functions for loading the dataset.
- **data_preprocessing.py**: Contains functions for imputing missing values, encoding categorical features, and scaling features.
- **utils.py**: Contains utility functions for plotting data distributions and value counts.
- **model.py**: Contains functions for training machine learning models.
- **evaluation.py**: Contains functions for evaluating the performance of the models.

## Setup Instructions

1. Clone the repository:
    ```sh
    git clone https://github.com/kaurrmanpreett/Loan-Eligibility-model-solution.git
    

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt

4. Place the dataset (`credit.csv`) in the `data/` directory.

## Usage

Run the main script to execute the full pipeline from data loading to model evaluation:
```sh
python main.py

## Dependencies
pandas
numpy
seaborn
matplotlib
scikit-learn

Install the dependencies using:
   ```sh
pip install -r requirements.txt

This project is licensed under the Apache License 2.0.

 
