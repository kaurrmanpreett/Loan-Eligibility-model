import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def impute_missing_values(df):
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    return df

def encode_categorical(df):
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
    return df

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def preprocess_data(df):
    df = impute_missing_values(df)
    df = df.drop('Loan_ID', axis=1)
    df = encode_categorical(df)
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    X_scaled = scale_features(X)
    return X_scaled, y
