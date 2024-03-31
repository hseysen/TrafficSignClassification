import os

if __name__ == "__main__":
    # It turns out each image is duplicated in the dataset for some reason.
    # Each duplicated image has _1_ in the filename.
    rootdir = os.path.join("datasets", "TrafficSign")

    # DATA directory:
    print("Going through DATA directory...")
    current_dir = os.path.join(rootdir, "DATA")
    for folder in os.listdir(current_dir):
        folderdir = os.path.join(current_dir, folder)
        for image in os.listdir(folderdir):
            imagepath = os.path.join(folderdir, image)
            if "_1_" in imagepath:
                print(f"Removing:\t{imagepath}")
                os.remove(imagepath)
    print("Cleaned DATA directory!")

    # TEST directory:
    print("Going through TEST directory...")
    current_dir = os.path.join(rootdir, "TEST")
    for image in os.listdir(current_dir):
        imagepath = os.path.join(current_dir, image)
        if "_1_" in imagepath:
            print(f"Removing:\t{imagepath}")
            os.remove(imagepath)
    print("Cleaned TEST directory!")
