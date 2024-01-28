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
    filenames_genuine = glob.glob('../../train/genuine/*')
    filenames_gan = glob.glob(f'../../train/{folder}/*')

    # create mean of dct coeffiecents of genuine images
    mean_dct_genuine = np.zeros(cv2.imread(filenames_genuine[0], cv2.IMREAD_GRAYSCALE).shape)
    for filename in filenames_genuine:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        # add 1e-10 to avoid division by zero
        mean_dct_genuine += np.log(np.abs(cv2.dct(np.float32(image))) + 1e-10)

    mean_dct_genuine /= len(filenames_genuine)

    # create mean of dct coefficients of gan images
    mean_dct_gan = np.zeros(cv2.imread(filenames_gan[0], cv2.IMREAD_GRAYSCALE).shape)
    for filename in filenames_gan:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        # add 1e-10 to avoid division by zero
        mean_dct_gan += np.log(np.abs(cv2.dct(np.float32(image))) + 1e-10)

    mean_dct_gan /= len(filenames_gan)

    # Needs to be repeated for each gan class and applied on corresponding val images!!!!
    fingerpint_peak = np.exp(mean_dct_gan - mean_dct_genuine)
    return fingerpint_peak


def apply_attack(s=100, t=0.1):
    for ds in datasets:
        os.chdir(f"{ds}/val_peak")
        # remove genuine and spoofed from removing fingerprints
        directories = os.listdir()
        directories.remove("genuine")
        directories.remove("spoofed")
        for method in directories:
            os.chdir(method)
            fingerprint_peak = determine_peak_fingerprint(method)
            # scale F_p to [0,1]
            fingerprint_peak = (fingerprint_peak - np.min(fingerprint_peak)) / (np.max(fingerprint_peak) - np.min(fingerprint_peak))
            # set values under threshold t to zero
            fingerprint_peak[fingerprint_peak < t] = 0
            fingerprint_peak *= s
            # scale F_p again to [0,1]
            fingerprint_peak = (fingerprint_peak - np.min(fingerprint_peak)) / (np.max(fingerprint_peak) - np.min(fingerprint_peak))
            for img in glob.glob("*"):
                image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                dct_result = cv2.dct(np.float32(image))
                modified_coeffs = dct_result * (1 - fingerprint_peak)
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
