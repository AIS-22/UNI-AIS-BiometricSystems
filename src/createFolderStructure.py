import os
import shutil
import splitfolders
import cv2


def delete_files_and_folders():
    # Iterate through all datasets and classes and delete .txt, .pdf files and resize folders
    for root, dirs, files in os.walk("."):
        # Delete if .txt or .pdf
        for file in files:
            if file.endswith(".txt") or file.endswith(".pdf") or file.endswith(".DS_Store"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
        # Delete folders containing the string "rs"
        for folder in dirs:
            if "_rs" in folder:
                folder_path = os.path.join(root, folder)
                shutil.rmtree(folder_path)


def structure_plus_dataset():
    # PLUS dataset has already the needed structure for genuine and spoofed.
    # Only the synthethic classes must be structured
    # class synthethic_stargan-v2 has for each fold two subfolders (latent and reference)
    # the folders have different images but the same name. By moving the images get overwritten
    # --> result are all images from the folder reference
    # all other synthethic classes have just the reference folder (690 latent images get lost)
    # ToDo: analyse if the latent images are important, if so rename them before moving
    print("PLUS dataset:")
    os.chdir("PLUS")

    # Delete all variants except 003 and 004
    for db_class in os.listdir("."):
        if os.path.isdir(db_class):
            os.chdir(db_class)
            for variant in os.listdir("."):
                if os.path.isdir(variant) and "synthethic" in db_class and variant not in ["003", "004"]:
                    shutil.rmtree(variant)
            os.chdir("..")
    print("All variants except 003 and 004 deleted")

    # Move images to variant folder
    for db_class in os.listdir("."):
        if os.path.isdir(db_class):
            os.chdir(db_class)
            for variant in os.listdir("."):
                if os.path.isdir(variant):
                    os.chdir(variant)
                    for fold in os.listdir("."):
                        if os.path.isdir(fold):
                            for subdir in os.listdir(fold):
                                for filename in os.listdir(os.path.join(fold, subdir)):
                                    source_path = os.path.join(fold, subdir, filename)
                                    # move file
                                    shutil.move(source_path, filename)
                            # remove empty folder
                            shutil.rmtree(fold)
                    os.chdir("..")
            os.chdir("..")
    print("Moved images from fold folder to variant folder and removed fold folders")
    os.chdir("..")


def structure_idiap_dataset():
    # class synthethic_stargan-v2 has for each fold two subfolders (latent and reference)
    # the folders have different images but the same name. By moving the images get overwritten
    # --> result are all images from the folder reference
    # all other synthethic classes have just the reference folder (380 latent images get lost)
    # ToDo: analyse if the latent images are important, if so rename them before moving
    print("IDIAP dataset:")
    os.chdir("IDIAP")

    # Delete all variants except 009
    for db_class in os.listdir("."):
        if os.path.isdir(db_class):
            os.chdir(db_class)
            for variant in os.listdir("."):
                if os.path.isdir(variant) and "synthethic" in db_class and variant != "009":
                    shutil.rmtree(variant)
            os.chdir("..")
    print("All variants except 009 deleted")

    # Move images to variant folder
    for db_class in os.listdir("."):
        if os.path.isdir(db_class):
            os.chdir(db_class)
            if "synthethic" in db_class:
                for variant in os.listdir("."):
                    if os.path.isdir(variant):
                        os.chdir(variant)
                        for fold in os.listdir("."):
                            if os.path.isdir(fold):
                                for subdir in os.listdir(fold):
                                    for filename in os.listdir(os.path.join(fold, subdir)):
                                        source_path = os.path.join(fold, subdir, filename)
                                        # move file
                                        shutil.move(source_path, filename)
                                # remove empty folder
                                shutil.rmtree(fold)
                        os.chdir("..")
            else:
                # genuine and spoofed structure differs from synthethic classes
                for subdir in os.listdir("."):
                    if os.path.isdir(subdir):
                        for filename in os.listdir(subdir):
                            source_path = os.path.join(subdir, filename)
                            # move file
                            shutil.move(source_path, filename)
                        # remove empty folder
                        shutil.rmtree(subdir)
            os.chdir("..")
    print("Moved images from fold folder to variant folder and removed fold folders")
    os.chdir("..")


def structure_protect_dataset():
    # class synthethic_stargan-v2 has for each fold two subfolders (latent and reference)
    # the folders have different images but the same name. By moving the images get overwritten
    # --> result are all images from the folder reference
    # all other synthethic classes have just the reference folder (228 latent images get lost)
    # ToDo: analyse if the latent images are important, if so rename them before moving
    print("PROTECT dataset:")
    os.chdir("PROTECT")

    # Move images to variant folder
    for db_class in os.listdir("."):
        if os.path.isdir(db_class):
            os.chdir(db_class)
            if "synthethic" in db_class:
                for variant in os.listdir("."):
                    if os.path.isdir(variant):
                        os.chdir(variant)
                        # PROTECT dataset has no fold directories for the variant 110
                        if "110" in variant:
                            for subdir in os.listdir("."):
                                if os.path.isdir(subdir):
                                    for filename in os.listdir(subdir):
                                        source_path = os.path.join(subdir, filename)
                                        # move file
                                        shutil.move(source_path, filename)
                                    # remove empty folder
                                    shutil.rmtree(subdir)
                        else:
                            for fold in os.listdir("."):
                                if os.path.isdir(fold):
                                    for subdir in os.listdir(fold):
                                        for filename in os.listdir(os.path.join(fold, subdir)):
                                            source_path = os.path.join(fold, subdir, filename)
                                            # move file
                                            shutil.move(source_path, filename)
                                    # remove empty folder
                                    shutil.rmtree(fold)
                        os.chdir("..")
            else:
                # genuine and spoofed structure differs from synthethic classes
                for subdir in os.listdir("."):
                    if os.path.isdir(subdir):
                        for filename in os.listdir(subdir):
                            source_path = os.path.join(subdir, filename)
                            # move file
                            shutil.move(source_path, filename)
                        # remove empty folder
                        shutil.rmtree(subdir)
            os.chdir("..")
    print("Moved images from fold folder to variant folder and removed fold folders")
    os.chdir("..")


def structure_scut_dataset():
    # class synthethic_stargan-v2 has for each fold two subfolders (latent and reference)
    # the folders have different images but the same name. By moving the images get overwritten
    # --> result are all images from the folder reference
    # all other synthethic classes have just the reference folder (690 latent images get lost)
    # ToDo: analyse if the latent images are important, if so rename them before moving
    print("SCUT dataset:")
    os.chdir("SCUT")

    # Delete all variants except 007 and 008
    for db_class in os.listdir("."):
        if os.path.isdir(db_class):
            os.chdir(db_class)
            for variant in os.listdir("."):
                if os.path.isdir(variant) and "synthethic" in db_class and variant not in ["007", "008"]:
                    shutil.rmtree(variant)
            os.chdir("..")
    print("All variants except 007 and 008 deleted")

    # Move images to variant folder
    for db_class in os.listdir("."):
        if os.path.isdir(db_class):
            os.chdir(db_class)
            if "synthethic" in db_class:
                for variant in os.listdir("."):
                    if os.path.isdir(variant):
                        os.chdir(variant)
                        for fold in os.listdir("."):
                            if os.path.isdir(fold):
                                for subdir in os.listdir(fold):
                                    for filename in os.listdir(os.path.join(fold, subdir)):
                                        source_path = os.path.join(fold, subdir, filename)
                                        # move file
                                        shutil.move(source_path, filename)
                                # remove empty folder
                                shutil.rmtree(fold)
                        os.chdir("..")
            else:
                # genuine and spoofed structure differs from synthethic classes
                for subdir in os.listdir("."):
                    if os.path.isdir(subdir):
                        for filename in os.listdir(subdir):
                            source_path = os.path.join(subdir, filename)
                            # move file
                            shutil.move(source_path, filename)
                        # remove empty folder
                        shutil.rmtree(subdir)
            os.chdir("..")
    print("Moved images from fold folder to variant folder and removed fold folders")
    os.chdir("..")


def create_class_for_each_variant():
    # each synthethic class has subfolder (variant). This function splits them into different classes for each variant
    datasets = ["PLUS", "SCUT", "PROTECT", "IDIAP"]
    for db in datasets:
        if os.path.isdir(db):
            os.chdir(db)
            # Copy each variant to its own folder
            for db_class in os.listdir("."):
                if "synthethic" in db_class:
                    for variant in os.listdir(db_class):
                        new_folder = os.path.join(db_class, variant).replace("/", "_")
                        shutil.copytree(db_class, new_folder)
                        if os.path.isdir(new_folder):
                            os.chdir(new_folder)
                            # delete all other variants in folder
                            for subdir in os.listdir("."):
                                if variant not in subdir and os.path.isdir(subdir):
                                    shutil.rmtree(subdir)
                            # move images out of variant folder
                            if os.path.isdir(variant):
                                for filename in os.listdir(variant):
                                    source_path = os.path.join(variant, filename)
                                    shutil.move(source_path, filename)

                                # delete empty folder
                                shutil.rmtree(variant)
                            os.chdir("..")
                    # delete folder with all variants in it
                    shutil.rmtree(db_class)
            os.chdir("..")

    print("Created for every variant own class")


def train_test_split():
    datasets = ["PLUS", "SCUT", "PROTECT", "IDIAP"]
    for db in datasets:
        # train test split
        splitfolders.ratio(db, output="../data/" + db,
                           seed=42, ratio=(.8, .2),
                           group_prefix=None, move=False)


def preprocess_scut_dataset():
    # Rotate the genuine and spoofed images by 90° to the right
    # get all folders in SCUT train and validation folder and write them in a list
    train_folders = ["genuine", "spoofed"]
    validation_folders = ["genuine", "spoofed"]
    # go through all genuine and spoofed images and rotate them by 90° to the right
    for folder in train_folders:
        for image in os.listdir(os.path.join("data/SCUT/train", folder)):
            image_path = os.path.join("data/SCUT/train", folder, image)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(image_path, image)
    for folder in validation_folders:
        for image in os.listdir(os.path.join("data/SCUT/val", folder)):
            image_path = os.path.join("data/SCUT/val", folder, image)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(image_path, image)
    print("Preprocessed SCUT dataset")


def resize_data_to_same_size(width=580, height=280):
    datasets = ["PLUS", "SCUT", "PROTECT", "IDIAP"]
    # copy the data folder and rename it to data_rs
    if not os.path.isdir("data_rs"):
        shutil.copytree("data", "data_rs")
    for db in datasets:
        # go trough all images and resize them to the same size
        for folder in os.listdir(os.path.join("data_rs", db, "train")):
            for image in os.listdir(os.path.join("data_rs", db, "train", folder)):
                image_path = os.path.join("data_rs", db, "train", folder, image)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(image_path, image)
        for folder in os.listdir(os.path.join("data_rs", db, "val")):
            for image in os.listdir(os.path.join("data_rs", db, "val", folder)):
                image_path = os.path.join("data_rs", db, "val", folder, image)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
                    cv2.imwrite(image_path, image)
        print(f"Resized {db} dataset to {width}x{height}")


def resize_data():
    datasets = ["PLUS", "SCUT", "PROTECT", "IDIAP"]
    for db in datasets:
        # get all folders in train and validation folder and write them in a list
        train_folders = []
        validation_folders = []
        for _, dirs, _ in os.walk(f"data/{db}/train"):
            for folder in dirs:
                train_folders.append(folder)
        for _, dirs, _ in os.walk(f"data/{db}/val"):
            for folder in dirs:
                validation_folders.append(folder)
        # remove the genuine and spoofed folder from the list
        train_folders.remove("genuine")
        train_folders.remove("spoofed")
        validation_folders.remove("genuine")
        validation_folders.remove("spoofed")
        # get of one image in the genuine folder the height and width
        genuine_folder = f"data/{db}/train/genuine"
        genuine_images = os.listdir(genuine_folder)
        if len(genuine_images) > 0:
            image_path = os.path.join(genuine_folder, genuine_images[0])
            image = cv2.imread(image_path)
            if image is not None:
                genuine_height, genuine_width, _ = image.shape
        # go through all the folders in the list if the image is not the same size resize it
        for folder in train_folders:
            for image in os.listdir(os.path.join(f"data/{db}/train", folder)):
                image_path = os.path.join(f"data/{db}/train", folder, image)
                image = cv2.imread(image_path)
                if image is not None:
                    height, width, _ = image.shape
                    if height != genuine_height or width != genuine_width:
                        image = cv2.resize(image, (genuine_width, genuine_height), interpolation=cv2.INTER_LANCZOS4)
                        cv2.imwrite(image_path, image)
        for folder in validation_folders:
            for image in os.listdir(os.path.join(f"data/{db}/val", folder)):
                image_path = os.path.join(f"data/{db}/val", folder, image)
                image = cv2.imread(image_path)
                if image is not None:
                    height, width, _ = image.shape
                    if height != genuine_height or width != genuine_width:
                        image = cv2.resize(image, (genuine_width, genuine_height), interpolation=cv2.INTER_LANCZOS4)
                        cv2.imwrite(image_path, image)
        print(f"Preprocessed {db} dataset")


def main():
    os.chdir("data_prepared")
    delete_files_and_folders()
    structure_scut_dataset()
    structure_idiap_dataset()
    structure_protect_dataset()
    structure_plus_dataset()
    create_class_for_each_variant()
    train_test_split()
    os.chdir("..")
    preprocess_scut_dataset()
    resize_data_to_same_size()  # create data_rs folder and resize data so all datasets match
    resize_data()  # resize data so one dataset matches internaly


if __name__ == '__main__':
    main()
