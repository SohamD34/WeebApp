import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, confusion_matrix,
    roc_auc_score, precision_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostClassifier, Pool
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


def print_cwd():
    return os.getcwd()


def load_dataframe():
    df = pd.read_csv('E:\Python Projects\WebApp\data\Customer-Churn.csv')
    print("Loaded the dataframe...\n")
    return df


def get_categorical_columns(df):
    return df.select_dtypes(include=['object']).columns.tolist()



def handle_columns(df):

    print("Preprocessing the columns...\n")
    df = df.drop(['customerID'], axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
    df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

    columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for column in columns_to_replace:
        df[column] = df[column].replace('No internet service', 'No')

    return df



def split_df(df):

    print("Splitting the dataframe...\n")

    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=64)
    train_index, test_index = next(strat_split.split(df, df["Churn"]))
    strat_train_set, strat_test_set = df.loc[train_index], df.loc[test_index]

    X_train = strat_train_set.drop("Churn", axis=1)
    y_train = strat_train_set["Churn"].copy()

    X_test = strat_test_set.drop("Churn", axis=1)
    y_test = strat_test_set["Churn"].copy()

    return X_train, y_train, X_test, y_test



def train_model(X_train, y_train, X_test, y_test):

    print("Training the model...\n")
    cat_model = CatBoostClassifier(verbose=False, random_state=0, scale_pos_weight=3)

    categorical_columns = get_categorical_columns(X_train)

    cat_model.fit(X_train, y_train, cat_features=categorical_columns, eval_set=(X_test, y_test))
    y_pred = cat_model.predict(X_train)

    return cat_model, y_pred



def get_metrics(y, predictions):
    
    accuracy = accuracy_score(y, predictions)
    recall = recall_score(y, predictions) 
    roc_auc = roc_auc_score(y, predictions) 
    precision = precision_score(y, predictions)
    result = pd.DataFrame({'Accuracy': accuracy, 'Recall': recall, 'Roc_Auc': roc_auc, 'Precision': precision}, index=['Catboost'])
    
    return result


def log(result):

    print("Logging the results...\n")
    log_dir = "../logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #------ Write the results to a log file ------#


def train():

    X_train, y_train, X_test, y_test = split_df(handle_columns(load_dataframe()))
    cat_model, train_predictions = train_model(X_train, y_train, X_test, y_test)
    result = get_metrics(y_train, train_predictions)

    print(result)
    print("\n"+os.getcwd())
    log(result)

    model_path = 'E:/Python Projects/WebApp/models/cat_model.pkl'
    pickle.dump(cat_model, open(model_path, 'wb'))


if __name__ == "__main__":
    train()