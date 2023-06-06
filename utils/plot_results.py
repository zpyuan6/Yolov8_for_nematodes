import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_performance():
    mAP_record = pd.read_csv("model_mAP.csv")

    sns.lineplot(y="mAP", x="Model", hue="dataset", style="dataset", data=mAP_record, markers="o")
    plt.show()


if __name__ == "__main__":
    plot_performance()
