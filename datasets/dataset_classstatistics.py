import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    rootdir = os.path.join("datasets", "TrafficSign")

    # Initialize class counts
    class_counts = np.array([0] * 58)               # Total number of classes

    current_dir = os.path.join(rootdir, "DATA")
    for folder in os.listdir(current_dir):
        folderdir = os.path.join(current_dir, folder)
        class_counts[int(folder)] = len(os.listdir(folderdir))
    class_ids = list(*np.where(class_counts != 0))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(class_ids, class_counts[class_counts != 0], color="skyblue")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(class_ids, rotation=70, ha="right")
    plt.yticks(range(0, max(class_counts) + 1, 20))
    plt.grid()
    plt.show()
