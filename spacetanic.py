#!/usr/bin/env python
import pandas as pd
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

    training_file = pd.read_csv("train.csv")
    training_file = derived_variables(training_file)
    training_file = remove_nan(training_file)
    if DO_FEATS_PLOT:
        plot_variables(training_file, "before_enc")
    training_file = encode_data(training_file)
    if DO_FEATS_PLOT:
        plot_variables(training_file, "after_enc")
    training_file = training_file.drop("Name", axis=1)
    data = training_file.astype("float64")
    X_train, X_test, Y_train, Y_test = split_train_test(data)
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

    Y = data["Transported"]
    X = data.drop("Transported", axis=1)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print("Number of test samples : ", X_test.shape[0])
    print("Number of train samples : ", X_train.shape[0])
    return X_train, X_test, Y_train, Y_test


def encode_data(data):
    """encoding string features to have numerical value"""

    no_encoded_feats = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "Number",
        "Group",
        "Single",
        "VRDeck",
        "Name",
        "Transported",
    ]

    enc_feats = [str(item) for item in data.keys() if item not in no_encoded_feats]
    data[enc_feats] = data[enc_feats].astype(str)
    enc = OrdinalEncoder()
    data[enc_feats] = enc.fit_transform(data[enc_feats])
    data[["Transported"]] = enc.fit_transform(data[["Transported"]])
    return data


def remove_nan(data):
    """replacing NaN values with -99"""
    df = data.copy()
    df = df.fillna(-99)

    return df


def derived_variables(data):
    """Takes Cabin and creates three variables
    Decomposes PassangerId into two variables"""

    Deck = data["Cabin"].str.split("/").str.get(0)
    Number = data["Cabin"].str.split("/").str.get(1)
    Side = data["Cabin"].str.split("/").str.get(2)
    data["Deck"], data["Number"], data["Side"] = (
        Deck,
        list(np.float_(Number)),
        Side,
    )
    Group = data["PassengerId"].str.split("_").str.get(0)
    Single = data["PassengerId"].str.split("_").str.get(1)
    data["Group"], data["Single"] = (
        list(np.float_(Group)),
        list(np.float_(Single)),
    )

    data = data.drop("Cabin", axis=1)
    data = data.drop("PassengerId", axis=1)
    return data


def plot_variables(data, suffix):
    """Takes every variable in the dataset
    and creates a plot comparing
    the case of true or false transported"""

    num_feats = [
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "Number",
        "Group",
        "VRDeck",
    ]

    fig, ax = plt.subplots()
    for keys in data.keys():
        if keys == "Name":
            continue
        print(f"Plotting variable... {keys}")
        if keys in num_feats:
            binwidth = 2
            if keys != "Age":
                binwidth = 1000
            if keys == "Number":
                binwidth = 100
            sns.histplot(data=data, x=keys, hue="Transported", binwidth=binwidth)
        else:
            sns.countplot(
                data=data,
                x=keys,
                hue="Transported",
                order=data[keys].value_counts().index,
            )
        plt.savefig(f"input_feat/{keys}_{suffix}.png")
        plt.cla()


if __name__ == "__main__":
    main()
