import numpy as np

import pandas as pd
import matplotlib.pyplot as plt


def loadData(path):
    return pd.read_csv(path)

def addColumns(df):
    #adding some non-linear connections between the features
    df["sepal area"] = df["sepal length (cm)"] * df["sepal width (cm)"]
    df["petal area"] = df["petal length (cm)"] * df["petal width (cm)"]
    df["sepal aspect"] = df["sepal length (cm)"] / df["sepal width (cm)"]
    df["petal aspect"] = df["petal length (cm)"] / df["petal width (cm)"]
    df["area ratio"] = df["sepal area"] / df["petal area"]
    df["sepal diagonal"] = np.sqrt(df["sepal length (cm)"]**2 + df["sepal width (cm)"]**2)
    df["petal diagonal"] = np.sqrt(df["petal length (cm)"]**2 + df["petal width (cm)"]**2)
    df["petal width X sepal length"] = df["petal width (cm)"] * df["sepal length (cm)"]
    df["petal length X sepal width"] = df["petal length (cm)"] * df["sepal width (cm)"]


def statistics(df):
    corr_matrix = df.corr()
    print(df.round(2))
    print(corr_matrix.round(2))
    corr_pairs = (
        corr_matrix.unstack()
            .reset_index()
            .rename(columns={"level_0": "var1", "level_1": "var2", 0: "correlation"})
    )
    corr_pairs = corr_pairs[corr_pairs["var1"] < corr_pairs["var2"]]
    corr_pairs["correlation_abs"] = corr_pairs["correlation"].abs()
    top5 = corr_pairs.nlargest(5,"correlation_abs").reset_index(drop=True).round(4)
    least5 = corr_pairs.nsmallest(5,"correlation_abs").reset_index(drop=True).round(4)
    print("Top 5 correlations are:")
    print("\n".join(f'{row.Index+1}. {row.var1} and {row.var2}: {row.correlation_abs}' for i,row in enumerate(top5.itertuples())))
    print("Least 5 correlations are:")
    print("\n".join(f'{row.Index+1}. {row.var1} and {row.var2}: {row.correlation_abs}' for i,row in enumerate(least5.itertuples())))
    #print_Corr_Bars(corr_pairs,top5, least5)


def print_Corr_Bars(data, top5, least5):
    xlabels = [f"{a} & {b}" for a,b in zip(data["var1"],data["var2"])]
    x = [i*0.6 for i in range(len(xlabels))]
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 8))

    corr_bar = plt.bar([i - (width / 2) for i in x], data["correlation"], width, label="Correlation")
    plt.bar([i + (width / 2) for i in x], data["correlation_abs"], width, label="Correlation_abs")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Value", fontsize=14)
    ax.legend(loc="upper center")
    ax.margins(x=0.01)
    for row in top5.itertuples():
        label = f"{row.var1} & {row.var2}"
        idx = xlabels.index(label)
        bar = corr_bar[idx]
        ax.text(bar.get_x() + width, bar.get_height() + 0.01, f'#{row.Index+1}', ha="center", va="bottom")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.title.set_text("Iris Correlation")
    plt.tight_layout()
    plt.show()



def normalize(df, test =None):
    #will be using minmax scaler
    normalized = df.copy()

    if test is None:
        for column in df.columns:
            temp_min = normalized[column].min()
            temp_max = normalized[column].max()
            denom = temp_max - temp_min
            if denom == 0:
                normalized[column] = 0.0
            else:
                normalized[column] = ((normalized[column] - temp_min)/ denom)
        return normalized
    else:
        test_Normalized = test.copy()
        for column in df.columns:
            temp_min = normalized[column].min()
            temp_max = normalized[column].max()
            denom = temp_max - temp_min
            if denom == 0:
                normalized[column] = 0.0
                test_Normalized[column] = 0.0
            else:
                normalized[column] = ((normalized[column] - temp_min) / denom)
                test_Normalized[column] = ((test_Normalized[column] - temp_min) / denom)
        return normalized, test_Normalized

def distance_matrix(df):
    x = df.to_numpy()
    distances = np.zeros((len(x), len(x)))

    for row1 in range(len(x)):
        curr_sample = x[row1] #getts a vector with 13 entries (sample)
        for row2 in range(row1 + 1 , len(x)):
            other_sample = x[row2]
            distances[row1, row2] = np.linalg.norm(curr_sample - other_sample)
    return distances