import os
import shutil
import cv2
import numpy as np
import glob

datasets = ["PLUS", "SCUT", "PROTECT", "IDIAP"]


def copy_images():
    for ds in datasets:
        os.chdir(ds)
        shutil.copytree(src="val", dst="val_peak", dirs_exist_ok=True)
        print(f"Copied {ds}")
        os.chdir("..")


def determine_peak_fingerprint(folder):
    filenames_genuine = glob.glob(f'../../train/genuine/*')
    filenames_gan = glob.glob(f'../../train/{folder}/*')

    # create mean of dct coeffiecents of genuine images
    mean_dct_genuine = np.zeros(cv2.imread(filenames_genuine[0], cv2.IMREAD_GRAYSCALE).shape)
    for filename in filenames_genuine:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        mean_dct_genuine += np.log(np.abs(cv2.dct(np.float32(image))))

    mean_dct_genuine /= len(filenames_genuine)

    # create mean of dct coefficients of gan images
    mean_dct_gan = np.zeros(cv2.imread(filenames_gan[0], cv2.IMREAD_GRAYSCALE).shape)
    for filename in filenames_gan:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        mean_dct_gan += np.log(np.abs(cv2.dct(np.float32(image))))

    mean_dct_gan /= len(filenames_gan)

    # Needs to be repeated for each gan class and applied on corresponding val images!!!!
    F_p = np.exp(mean_dct_gan - mean_dct_genuine)
    return F_p


def apply_attack(s=100, t=0.1):
    for ds in datasets:
        os.chdir(f"{ds}/val_peak")
        # remove genuine foler
        shutil.rmtree("genuine")
        shutil.rmtree("spoofed")
        for method in os.listdir():
            os.chdir(method)
            F_p = determine_peak_fingerprint(method)
            # scale F_p to [0,1]
            F_p = (F_p - np.min(F_p)) / (np.max(F_p) - np.min(F_p))
            # set values under threshold t to zero
            F_p[F_p < t] = 0
            F_p *= s
            # scale F_p again to [0,1]
            F_p = (F_p - np.min(F_p)) / (np.max(F_p) - np.min(F_p))
            for img in glob.glob("*"):
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                dct_result = cv2.dct(np.float32(image))
                modified_coeffs = dct_result * (1 - F_p)
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
