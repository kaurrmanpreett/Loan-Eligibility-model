import seaborn as sns
import matplotlib.pyplot as plt


def plot_value_counts(df, column):
    df[column].value_counts().plot.bar()
    plt.show()

def plot_distribution(df, column):
    sns.distplot(df[column])
    plt.show()
