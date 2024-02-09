import os
import shutil
import cv2
import numpy as np
import glob

datasets = ["PLUS", "SCUT", "PROTECT", "IDIAP"]


def copy_images():
    for ds in datasets:
        os.chdir(ds)
        shutil.copytree(src="val", dst="val_mean", dirs_exist_ok=True)
        print(f"Copied {ds}")
        os.chdir("..")


def determine_mean_fingerprint(folder):
    filenames_genuine = glob.glob('../../train/genuine/*')
    filenames_gan = glob.glob(f'../../train/{folder}/*')

    # create mean of dct coefficients genuine images
    mean_dct_genuine = np.zeros(cv2.imread(filenames_genuine[0], cv2.IMREAD_GRAYSCALE).shape)
    for filename in filenames_genuine:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        mean_dct_genuine += cv2.dct(np.float32(image))

    mean_dct_genuine /= len(filenames_genuine)

    # create mean of dct coeffiecents of gan images
    mean_dct_gan = np.zeros(cv2.imread(filenames_gan[0], cv2.IMREAD_GRAYSCALE).shape)
    for filename in filenames_gan:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        mean_dct_gan += cv2.dct(np.float32(image))

    mean_dct_gan /= len(filenames_gan)

    # Needs to be repeated for each gan class and applied on corresponding val images!!!!
    fingerprint_mean = mean_dct_gan - mean_dct_genuine
    return fingerprint_mean


def apply_attack(s=0.5):
    for ds in datasets:
        os.chdir(f"{ds}/val_mean")
        # remove genuine and spoofed from removing fingerprints
        directories = os.listdir()
        directories.remove("genuine")
        directories.remove("spoofed")
        for method in directories:
            os.chdir(method)
            fingerprint_mean = determine_mean_fingerprint(method)
            for img in glob.glob("*"):
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                dct_result = cv2.dct(np.float32(image))
                modified_coeffs = dct_result - (s * fingerprint_mean)
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
