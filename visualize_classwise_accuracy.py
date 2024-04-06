import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("classwise_accuracy.csv")

class_names = data["Class Name"].tolist()
true_positives = data["True Positives"].tolist()
false_positives = data["False Positives"].tolist()

fig, axes = plt.subplots(nrows=11, ncols=5, figsize=(20, 40))
axes = axes.flatten()

for i, (class_name, tp, fp) in enumerate(zip(class_names, true_positives, false_positives)):
    labels = ["True Positives", "False Positives"]
    sizes = [tp, fp]
    ax = axes[i]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title(class_name)

plt.tight_layout()

plt.savefig("all_pie_charts.png")
