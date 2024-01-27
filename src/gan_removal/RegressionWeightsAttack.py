import os
import shutil
import cv2
import numpy as np
import glob
from sklearn.linear_model import Lasso
import random

random.seed(42)
datasets = ["PLUS", "SCUT", "PROTECT", "IDIAP"]


def copy_images():
    for ds in datasets:
        os.chdir(ds)
        shutil.copytree(src="val", dst="val_reg", dirs_exist_ok=True)
        print(f"Copied {ds}")
        os.chdir("..")


def determine_regression_parameter(folder):
    filenames_genuine = glob.glob(f'../../train/genuine/*')
    filenames_gan = glob.glob(f'../../train/{folder}/*')

    # create mean of dct coefficients genuine images
    lasso_model = Lasso(alpha=0.1)
    for filenames_gan in filenames_gan:
        for filename in random.sample(filenames_genuine, 10):
            image_gen = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            image_gan = cv2.imread(filenames_gan, cv2.IMREAD_GRAYSCALE)
            dct_result_gen = cv2.dct(np.float32(image_gen))
            dct_result_gan = cv2.dct(np.float32(image_gan))
            lasso_model.fit(dct_result_gan, dct_result_gen)

    F_r = lasso_model.coef_
    return F_r


def apply_attack(s=50):
    for ds in datasets:
        os.chdir(f"{ds}/val_reg")
        # remove genuine foler
        shutil.rmtree("genuine")
        shutil.rmtree("spoofed")
        for method in os.listdir():
            os.chdir(method)
            F_r = s * determine_regression_parameter(method)
            # scale F_r to [-1,1]
            F_r = 2 * (F_r - np.min(F_r)) / (np.max(F_r) - np.min(F_r)) - 1
            for img in glob.glob("*"):
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                dct_result = cv2.dct(np.float32(image))
                modified_coeffs = dct_result * (1 - F_r)
                # store the new image in removal folder
                reconstructed_image = cv2.idct(modified_coeffs)
                cv2.imwrite(img, reconstructed_image)
            os.chdir("..")

        print(f"Finished {ds}")
        os.chdir("../..")


if __name__ == '__main__':
    os.chdir("data_rs")
    copy_images()
    apply_attack()
