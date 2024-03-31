import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    rootdir = os.path.join("datasets", "TrafficSign")

    # Initialize class counts
    class_counts = np.array([0] * 58)               # Total number of classes

    # Checking TEST directory
    current_dir = os.path.join(rootdir, "TEST")
    for image in os.listdir(current_dir):
        class_counts[int(image[:3])] += 1
    zero_test_cases = list(*np.where(class_counts == 0))
    print("Classes that have no test instances:", zero_test_cases)
    
    print("Removing training images corresponding to these classes")
    current_dir = os.path.join(rootdir, "DATA")
    for folder in zero_test_cases:
        folderdir = os.path.join(current_dir, str(folder))
        if os.path.exists(folderdir):
            for image in os.listdir(folderdir):
                imagedir = os.path.join(folderdir, image)
                print(f"Removing:\t{imagedir}")
                os.remove(imagedir)
            print(f"Removing:\t{folderdir}")
            os.rmdir(folderdir)      
