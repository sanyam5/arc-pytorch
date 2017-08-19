import os
import urllib.request
import numpy as np
import zipfile

from scipy.ndimage import imread


omniglot_url = 'http://github.com/brendenlake/omniglot/archive/master.zip'
data_dir = os.path.join("data")
zip_location = os.path.join(data_dir, "omniglot.zip")
unzip_location = os.path.join(data_dir, "extracted")
zipped_images_location = os.path.join(unzip_location, "omniglot-master", "python")
extracted_images_location = os.path.join(data_dir, "images")


def download() -> None:
    if os.path.exists(zip_location) and os.path.isfile(zip_location):
        print("File {} already exists. Skipping download.".format(zip_location))
        return
    print("Downloading the zip file from url {} and writing to {}".format(
        omniglot_url, zip_location
    ))
    urllib.request.urlretrieve(omniglot_url, zip_location)
    print("Finished downloading.")


def extract() -> None:
    print("Extracting {} to {}".format(zip_location, unzip_location))
    zip_ref = zipfile.ZipFile(zip_location, 'r')
    zip_ref.extractall(unzip_location)
    zip_ref.close()
    print("Finished extracting.")


def extract_images() -> None:
    image_sets = ["images_background.zip", "images_evaluation.zip"]
    image_sets = [os.path.join(zipped_images_location, image_set) for image_set in image_sets]
    print("Extracting image sets {}".format(image_sets))

    for image_set in image_sets:
        zip_ref = zipfile.ZipFile(image_set, 'r')
        zip_ref.extractall(extracted_images_location)
        zip_ref.close()

    print("Done extracting image sets.")


def omniglot_folder_to_NDarray(path_im):
    alphbts = os.listdir(path_im)
    ALL_IMGS = []

    for alphbt in alphbts:
        chars = os.listdir(os.path.join(path_im, alphbt))
        for char in chars:
            img_filenames = os.listdir(os.path.join(path_im, alphbt, char))
            char_imgs = []
            for img_fn in img_filenames:
                fn = os.path.join(path_im, alphbt, char, img_fn)
                I = imread(fn)
                I = np.invert(I)
                char_imgs.append(I)
            ALL_IMGS.append(char_imgs)

    return np.array(ALL_IMGS)


def save_to_numpy() -> None:
    image_folders = ["images_background", "images_evaluation"]
    all_np_array = []
    for image_folder in image_folders:
        np_array_loc = os.path.join(data_dir, image_folder + ".npy")
        print("Converting folder {} to numpy array...".format(image_folder))
        np_array = omniglot_folder_to_NDarray(os.path.join(extracted_images_location, image_folder))
        np.save(np_array_loc, np_array)
        all_np_array.append(np_array)
        print("Done.")

    all_np_array = np.concatenate(all_np_array, axis=0)
    np.save(os.path.join("data", "omniglot.npy"), all_np_array)


def main():
    download()
    extract()
    extract_images()
    save_to_numpy()

if __name__ == "__main__":
    main()
