from modules.data_loader import *
from modules.data_preprocessing import *
from modules.model import *
from modules.evaluation import *
from modules.utils import *

def main():
    data_path = 'data/credit.csv'
    df = load_data(data_path)
    
    # Initial EDA 
    plot_value_counts(df, 'Loan_Status')
    plot_distribution(df, 'LoanAmount')
    
    X, y = preprocess_data(df)
    
    # Train and evaluate Logistic Regression model
    lr_model, X_test, y_test = train_linear_regression(X, y)
    lr_accuracy, lr_conf_matrix, lr_report = evaluate_model(lr_model, X_test, y_test)
    print(f'Logistic Regression Accuracy: {lr_accuracy}')
    print(f'Logistic Regression Confusion Matrix:\n{lr_conf_matrix}')
    print(f'Logistic Regression Classification Report:\n{lr_report}')
    
    # Train and evaluate Random Forest model
    rf_model, X_test, y_test = train_random_forest(X, y)
    rf_accuracy, rf_conf_matrix, rf_report = evaluate_model(rf_model, X_test, y_test)
    print(f'Random Forest Accuracy: {rf_accuracy}')
    print(f'Random Forest Confusion Matrix:\n{rf_conf_matrix}')
    print(f'Random Forest Classification Report:\n{rf_report}')

if __name__ == "__main__":
    main()
