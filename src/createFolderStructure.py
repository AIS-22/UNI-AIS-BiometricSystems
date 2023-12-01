import os
import shutil
import splitfolders


def deleteFilesAndFolders():
    # Iterate through all datasets and classes and delete .txt, .pdf files and resize folders
    for root, dirs, files in os.walk("."):
        # Delete if .txt or .pdf
        for file in files:
            if file.endswith(".txt") or file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
        # Delete folders containing the string "rs"
        for folder in dirs:
            if "_rs" in folder:
                folder_path = os.path.join(root, folder)
                shutil.rmtree(folder_path)


def structurePLUSdataset():
    # PLUS dataset has already the needed structure for genuine and spoofed. Only the synthethic classes must be structured
    # class synthethic_stargan-v2 has for each fold two subfolders (latent and reference)
    # the folders have different images but the same name. By moving the images get overwritten
    # --> result are all images from the folder reference
    # all other synthethic classes have just the reference folder (690 latent images get lost)
    # ToDo: analyse if the latent images are important, if so rename them before moving
    print("PLUS dataset:")
    os.chdir("PLUS")

    # Delete all variants except 003 and 004
    for dbClass in os.listdir("."):
        if os.path.isdir(dbClass):
            os.chdir(dbClass)
            for variant in os.listdir("."):
                if os.path.isdir(variant) and "synthethic" in dbClass and variant not in ["003", "004"]:
                    shutil.rmtree(variant)
            os.chdir("..")
    print("All variants except 003 and 004 deleted")

    # Move images to variant folder
    for dbClass in os.listdir("."):
        os.chdir(dbClass)
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


def structureIDIAPdataset():
    # class synthethic_stargan-v2 has for each fold two subfolders (latent and reference)
    # the folders have different images but the same name. By moving the images get overwritten
    # --> result are all images from the folder reference
    # all other synthethic classes have just the reference folder (380 latent images get lost)
    # ToDo: analyse if the latent images are important, if so rename them before moving
    print("IDIAP dataset:")
    os.chdir("IDIAP")

    # Delete all variants except 009
    for dbClass in os.listdir("."):
        if os.path.isdir(dbClass):
            os.chdir(dbClass)
            for variant in os.listdir("."):
                if os.path.isdir(variant) and "synthethic" in dbClass and variant != "009":
                    shutil.rmtree(variant)
            os.chdir("..")
    print("All variants except 009 deleted")

    # Move images to variant folder
    for dbClass in os.listdir("."):
        os.chdir(dbClass)
        if "synthethic" in dbClass:
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


def structurePROTECTdataset():
    # class synthethic_stargan-v2 has for each fold two subfolders (latent and reference)
    # the folders have different images but the same name. By moving the images get overwritten
    # --> result are all images from the folder reference
    # all other synthethic classes have just the reference folder (228 latent images get lost)
    # ToDo: analyse if the latent images are important, if so rename them before moving
    print("PROTECT dataset:")
    os.chdir("PROTECT")

    # Move images to variant folder
    for dbClass in os.listdir("."):
        os.chdir(dbClass)
        if "synthethic" in dbClass:
            for variant in os.listdir("."):
                if os.path.isdir(variant):
                    os.chdir(variant)
                    # PROTECT dataset has no fold directories for the variant 110
                    if "110" in variant:
                        for subdir in os.listdir("."):
                            p = os.getcwd()
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


def structureSCUTdataset():
    # class synthethic_stargan-v2 has for each fold two subfolders (latent and reference)
    # the folders have different images but the same name. By moving the images get overwritten
    # --> result are all images from the folder reference
    # all other synthethic classes have just the reference folder (690 latent images get lost)
    # ToDo: analyse if the latent images are important, if so rename them before moving
    print("SCUT dataset:")
    os.chdir("SCUT")

    # Delete all variants except 007 and 008
    for dbClass in os.listdir("."):
        if os.path.isdir(dbClass):
            os.chdir(dbClass)
            for variant in os.listdir("."):
                if os.path.isdir(variant) and "synthethic" in dbClass and variant not in ["007", "008"]:
                    shutil.rmtree(variant)
            os.chdir("..")
    print("All variants except 007 and 008 deleted")

    # Move images to variant folder
    for dbClass in os.listdir("."):
        os.chdir(dbClass)
        if "synthethic" in dbClass:
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


def createClassForEachVariant():
    # each synthethic class has subfolder (variant). This function splits them into different classes for each variant
    datasets = ["PLUS", "SCUT", "PROTECT", "IDIAP"]
    for db in datasets:
        if os.path.isdir(db):
            os.chdir(db)
            # Copy each variant to its own folder
            for dbClass in os.listdir("."):
                if "synthethic" in dbClass:
                    for variant in os.listdir(dbClass):
                        new_folder = os.path.join(dbClass, variant).replace("/", "_")
                        shutil.copytree(dbClass, new_folder)
                        os.chdir(new_folder)
                        # delete all other variants in folder
                        for subdir in os.listdir("."):
                            if not variant in subdir and os.path.isdir(subdir):
                                shutil.rmtree(subdir)
                        # move images out of variant folder
                        for filename in os.listdir(variant):
                            source_path = os.path.join(variant, filename)
                            shutil.move(source_path, filename)

                        # delete empty folder
                        shutil.rmtree(variant)
                        os.chdir("..")
                    # delete folder with all variants in it
                    shutil.rmtree(dbClass)
            os.chdir("..")

    print("Created for every variant own class")


def trainTestSplit():
    datasets = ["PLUS", "SCUT", "PROTECT", "IDIAP"]
    for db in datasets:
        # train test split
        splitfolders.ratio(db, output="../data/" + db,
                           seed=42, ratio=(.8, .2),
                           group_prefix=None, move=False)


def main():
    os.chdir("data_prepared")
    deleteFilesAndFolders()
    structureSCUTdataset()
    structureIDIAPdataset()
    structurePROTECTdataset()
    structurePLUSdataset()
    createClassForEachVariant()
    trainTestSplit()
    os.chdir("..")


if __name__ == '__main__':
    main()
