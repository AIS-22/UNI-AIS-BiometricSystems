import os
import shutil
import cv2
import numpy as np
import glob

datasets = ["IDIAP"]


def copy_images():
    for ds in datasets:
        os.chdir(ds)
        shutil.copytree(src="val", dst="val_bar", dirs_exist_ok=True)
        print(f"Copied {ds}")
        os.chdir("..")


def copy_images_train():
    for ds in datasets:
        os.chdir(ds)
        shutil.copytree(src="train", dst="train_bar", dirs_exist_ok=True)
        print(f"Copied {ds}")
        os.chdir("..")


def apply_attack(s=20):
    for ds in datasets:
        os.chdir(f"{ds}/val_bar")
        # remove genuine and spoofed from removing fingerprints
        directories = os.listdir()
        directories.remove("genuine")
        directories.remove("spoofed")
        if ".DS_Store" in directories:
            directories.remove(".DS_Store")
        for method in directories:
            os.chdir(method)
            for img in glob.glob("*"):
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (666, 250), interpolation=cv2.INTER_LANCZOS4)
                dct_result = cv2.dct(np.float32(image))
                # set the last s rows and columns to zero
                dct_result[-s:, :] = 0
                dct_result[:, -s:] = 0
                reconstructed_image = cv2.idct(dct_result)
                reconstructed_image = cv2.resize(reconstructed_image, (665, 250), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(img, reconstructed_image)
            os.chdir("..")

        print(f"Finished {ds}")
        os.chdir("../..")


def apply_attack_train(s=20):
    for ds in datasets:
        os.chdir(f"{ds}/train_bar")
        # remove genuine and spoofed from removing fingerprints
        directories = os.listdir()
        directories.remove("genuine")
        directories.remove("spoofed")
        if ".DS_Store" in directories:
            directories.remove(".DS_Store")
        for method in directories:
            os.chdir(method)
            for img in glob.glob("*"):
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (666, 250), interpolation=cv2.INTER_LANCZOS4)
                dct_result = cv2.dct(np.float32(image))
                # set the last s rows and columns to zero
                dct_result[-s:, :] = 0
                dct_result[:, -s:] = 0
                reconstructed_image = cv2.idct(dct_result)
                reconstructed_image = cv2.resize(reconstructed_image, (665, 250), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(img, reconstructed_image)
            os.chdir("..")

        print(f"Finished {ds}")
        os.chdir("../..")


if __name__ == '__main__':
    os.chdir("data")
    copy_images()
    apply_attack()
    copy_images_train()
    apply_attack_train()
