#!/usr/bin/env python
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


DO_FEATS_PLOT = False

def main():
    """`main` function for spacetanic module"""

    train_data = pl.read_csv("train.csv")
    train_data = derived_variables(train_data)
    feat_type_dict = columns_types(train_data)
    train_data = remove_nan(train_data,feat_type_dict)
    if DO_FEATS_PLOT:
        plot_variables(train_data, feat_type_dict, "before_enc")
    train_data = encode_data(train_data,feat_type_dict)
    if DO_FEATS_PLOT:
        plot_variables(train_data, feat_type_dict, "after_enc")
    train_data = train_data.drop("Name")
    X_train, X_test, Y_train, Y_test = split_train_test(train_data)
    run_XGBoost(X_train, X_test, Y_train, Y_test)


def run_XGBoost(X_train, X_test, Y_train, Y_test):
    """defines XGBoost classifier and runs prediction"""

    XGB = XGBClassifier(n_estimators=500, learning_rate=0.05, early_stopping_rounds=5)
    XGB.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=False)
    y_pred_XGB = XGB.predict(X_test)
    XGB_accuracy = accuracy_score(Y_test, y_pred_XGB)
    print(f"Model accuracy is {XGB_accuracy}")


def split_train_test(data):
    """Takes dataset and splits into training and test"""

    #scikit-learn does not take polars DataFrame
    Y = data["Transported"].to_numpy()
    X = data.drop("Transported").to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print("Number of test samples : ", X_test.shape[0])
    print("Number of train samples : ", X_train.shape[0])
    return X_train, X_test, Y_train, Y_test



def columns_types(data):
    """Takes the dataframe and checks the dtype of columns"""

    enc_feats_string = []
    enc_feats_bool = []
    enc_feats_num = []
    feat_type_dic = {}

    dict_schema = data.schema
    for col, col_type in dict_schema.items():
        if isinstance(col_type, pl.String):
            enc_feats_string.append(col)
        elif isinstance(col_type, pl.Boolean):
            enc_feats_bool.append(col)
        else:
            enc_feats_num.append(col)

    feat_type_dic = {"num":enc_feats_num, 
                     "string": enc_feats_string, 
                     "bool":enc_feats_bool}

    return feat_type_dic

def encode_data(data,feat_type_dic):
    """encoding string features to have numerical value"""

    enc = OrdinalEncoder()
    data[feat_type_dic["string"]] = enc.fit_transform(data[feat_type_dic["string"]])
    data[feat_type_dic["bool"]]   = enc.fit_transform(data[feat_type_dic["bool"]])

    return data


def remove_nan(data,feat_type_dict):
    """replacing NaN values with -99"""
    #data[feat_type_dict["num"]] = data.fill_nan(-99)
    data = data.with_columns(
        pl.col(feat_type_dict["num"])
        .fill_nan(-99)
    )
    data = data.with_columns(
        pl.col(feat_type_dict["num"])
        .fill_null(-99)
    )
    data = data.with_columns(
        pl.col(feat_type_dict["string"])
        .fill_null("-99")
    )
    return data


def derived_variables(data):
    """Takes Cabin and creates three variables
    Decomposes PassangerId into three variables"""

    data = data.with_columns(
        [
            pl.col("Cabin")
            .str.split_exact("/", 2)
            .struct.rename_fields(["Deck", "Number", "Side"])
            .alias("fields"),
        ]
    ).unnest("fields")

    data = data.with_columns(
        [
            pl.col("PassengerId")
            .str.split_exact("_", 1)
            .struct.rename_fields(["Group", "Single"])
            .alias("fields"),
        ]
    ).unnest("fields")

    data = data.drop(["Cabin","PassengerId"])
    
    return data


def plot_variables(data, feat_type_dict, suffix):
    """Takes every variable in the dataset
    and creates a plot comparing
    the case of true or false transported"""

    num_feats = feat_type_dict["num"]

    fig, ax = plt.subplots()
    for col in data.columns:
        if col == "Name":
            continue
        print(f"Plotting variable... {col}")
        if col in num_feats:
            binwidth = 2
            if col != "Age":
                binwidth = 1000
            if col == "Number":
                binwidth = 100
            sns.histplot(data=data, x=col, hue="Transported", binwidth=binwidth)
        else:
            sns.countplot(
                data=data,
                x=col,
                hue="Transported"
            )
        plt.savefig(f"input_feat/{keys}_{suffix}.png")
        plt.cla()


if __name__ == "__main__":
    main()
