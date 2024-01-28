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


def determine_regression_parameter(folder, alpha=0.1, max_iter=1000, tol=0.1):
    filenames_genuine = glob.glob('../../train/genuine/*')
    filenames_gan = glob.glob(f'../../train/{folder}/*')

    # create mean of dct coefficients genuine images
    lasso_model = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)
    for filenames_gan in filenames_gan:
        for filename in random.sample(filenames_genuine, 10):
            image_gen = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            image_gan = cv2.imread(filenames_gan, cv2.IMREAD_GRAYSCALE)
            dct_result_gen = cv2.dct(np.float32(image_gen))
            dct_result_gan = cv2.dct(np.float32(image_gan))
            lasso_model.fit(dct_result_gan, dct_result_gen)

    return np.resize(lasso_model.coef_,image_gen.shape)


def apply_attack(s=50):
    for ds in datasets:
        os.chdir(f"{ds}/val_reg")
        # remove genuine foler
        shutil.rmtree("genuine")
        shutil.rmtree("spoofed")
        for method in os.listdir():
            os.chdir(method)
            fingerprint_regression_weights = s * determine_regression_parameter(method)
            # scale fingerprint_regression_weights to [-1,1]
            fingerprint_regression_weights = 2 * (fingerprint_regression_weights - np.min(fingerprint_regression_weights)) / (np.max(fingerprint_regression_weights) - np.min(fingerprint_regression_weights)) - 1
            for img in glob.glob("*"):
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                dct_result = cv2.dct(np.float32(image))
                modified_coeffs = dct_result * (1 - fingerprint_regression_weights)
                # store the new image in removal folder
                reconstructed_image = cv2.idct(modified_coeffs)
                cv2.imwrite(img, reconstructed_image)
            os.chdir("..")

        print(f"Finished {ds}")
        os.chdir("../..")


if __name__ == '__main__':
    os.chdir("data")
    copy_images()
    apply_attack()
