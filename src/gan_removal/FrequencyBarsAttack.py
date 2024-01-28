import os
import shutil
import cv2
import numpy as np
import glob

datasets = ["PLUS", "SCUT", "PROTECT", "IDIAP"]


def copy_images():
    for ds in datasets:
        os.chdir(ds)
        shutil.copytree(src="val", dst="val_bar", dirs_exist_ok=True)
        print(f"Copied {ds}")
        os.chdir("..")


def apply_attack(s=20):
    for ds in datasets:
        os.chdir(f"{ds}/val_bar")
        # remove genuine and spoofed from removing fingerprints
        directories = os.listdir()
        directories.remove("genuine")
        directories.remove("spoofed")
        for method in directories:
            os.chdir(method)
            for img in glob.glob("*"):
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                dct_result = cv2.dct(np.float32(image))
                # set the last s rows and columns to zero
                dct_result[-s:, :] = 0
                dct_result[:, -s:] = 0
                reconstructed_image = cv2.idct(dct_result)
                cv2.imwrite(img, reconstructed_image)
            os.chdir("..")

        print(f"Finished {ds}")
        os.chdir("../..")


if __name__ == '__main__':
    os.chdir("data")
    copy_images()
    apply_attack()
