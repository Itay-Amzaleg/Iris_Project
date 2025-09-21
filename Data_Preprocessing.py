import pandas as pd
import matplotlib.pyplot as plt


def loadData(path):
    return pd.read_csv(path)

def addColumns(df):
    df["is_big"] = df.apply(
        lambda row: 1 if (row["sepal length (cm)"] > 3 and row["sepal width (cm)"] > 3) else 0, axis=1)
    df["sepal avg"] = df[["sepal length (cm)", "sepal width (cm)"]].mean(axis=1)
    df["petal avg"] = (df["petal length (cm)"] + df["petal width (cm)"]) / 2.0

def statistics(df):
    statistics = df.describe().round(2)
    corr_matrix = df.corr()
    print(statistics)
    print(corr_matrix)
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



def normalize(df):
    #will be using minmax scaler
    normalized = df
    for column in df.columns:
        normalized[column] = ((normalized[column] - normalized[column].min())/ (normalized[column].max() - normalized[column].min()))
    return normalized