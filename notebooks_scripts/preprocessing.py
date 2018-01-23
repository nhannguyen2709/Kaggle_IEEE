# -*- coding: utf-8 -*-
"""
image preprocessing functions based on the OpenCV library 
"""
import numpy as np
import os
import cv2
from sklearn.feature_extraction.image import PatchExtractor
from shutil import copy

raw_imgs_dir = "/mnt/safe01/data/raw/"
processed_imgs_dir = "/mnt/safe01/data/processed/"
train_dir = "/mnt/safe01/data/train/"
validation_dir = "/mnt/safe01/data/validation/"
categories = sorted(os.listdir(raw_imgs_dir))

# JPEG compression
def jpeg_compress_70(path):
    img = cv2.imread(path)
    result, encoded_img = cv2.imencode(
        '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    decoded_img = cv2.imdecode(encoded_img, 1)
    return decoded_img


def jpeg_compress_90(path):
    img = cv2.imread(path)
    result, encoded_img = cv2.imencode(
        '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    decoded_img = cv2.imdecode(encoded_img, 1)
    return decoded_img


# resize via bicubic interpolation
def resize_05(path):
    img = cv2.imread(path)
    resized_img = cv2.resize(img, None,
                             fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_CUBIC)
    return resized_img


def resize_08(path):
    img = cv2.imread(path)
    resized_img = cv2.resize(img, None,
                             fx=0.8, fy=0.8,
                             interpolation=cv2.INTER_CUBIC)
    return resized_img


def resize_15(path):
    img = cv2.imread(path)
    resized_img = cv2.resize(img, None,
                             fx=1.5, fy=1.5,
                             interpolation=cv2.INTER_CUBIC)
    return resized_img


def resize_20(path):
    img = cv2.imread(path)
    resized_img = cv2.resize(img, None,
                             fx=2.0, fy=2.0,
                             interpolation=cv2.INTER_CUBIC)
    return resized_img

# gamma correction
def gamma_correction_08(path):
    img = cv2.imread(path)
    inv_gamma = 1.0 / 0.8
    lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255
                             for i in np.arange(0, 256)]).astype('uint8')
    corrected_img = cv2.LUT(img, lookup_table)
    return corrected_img


def gamma_correction_12(path):
    img = cv2.imread(path)
    inv_gamma = 1.0 / 1.2
    lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255
                             for i in np.arange(0, 256)]).astype('uint8')
    corrected_img = cv2.LUT(img, lookup_table)
    return corrected_img

        
def augment_training_img(input_path=raw_imgs_dir, output_path=processed_imgs_dir):
    """
    Augment 2750 original training images with all eight image alterations. 
    """
    for i in range(len(categories)):
        print("Started augmenting training images taken by " + categories[i] + ".")
        for img_name in sorted(os.listdir(input_path + categories[i] + '/')):
            srcfile = raw_imgs_dir + categories[i] + '/' + img_name
            dstroot = processed_imgs_dir + categories[i] + '/'
            copy(srcfile, dstroot)
            
            jpeg70_img = jpeg_compress_70(
                input_path + categories[i] + '/' + img_name)
            cv2.imwrite(output_path +
                        categories[i] + '/' + 
                        img_name.replace('unalt_', 'compressed70_'), jpeg70_img)

            jpeg90_img = jpeg_compress_90(
                input_path + categories[i] + '/' + img_name)
            cv2.imwrite(output_path +
                        categories[i] + '/' + 
                        img_name.replace('unalt_', 'compressed90_'), jpeg90_img)

            resized05_img = resize_05(
                input_path + categories[i] + '/' + img_name)
            cv2.imwrite(output_path +
                        categories[i] + '/' +
                        img_name.replace('unalt_', 'resized05_'), resized05_img)

            resized08_img = resize_08(
                input_path + categories[i] + '/' + img_name)
            cv2.imwrite(output_path +
                        categories[i] + '/' + 
                        img_name.replace('unalt_', 'resized08_'), resized08_img)

            resized15_img = resize_15(
                input_path + categories[i] + '/' + img_name)
            cv2.imwrite(output_path +
                        categories[i] + '/' + 
                        img_name.replace('unalt_', 'resized15_'), resized15_img)

            resized20_img = resize_20(
                input_path + categories[i] + '/' + img_name)
            cv2.imwrite(output_path +
                        categories[i] + '/' +
                        img_name.replace('unalt_', 'resized20_'), resized20_img)

            gamma08_img = gamma_correction_08(
                input_path + categories[i] + '/' + img_name)
            cv2.imwrite(output_path +
                        categories[i] + '/' + 
                        img_name.replace('unalt_', 'gamma08_'), gamma08_img)

            gamma12_img = gamma_correction_12(
                input_path + categories[i] + '/' + img_name)
            cv2.imwrite(output_path +
                        categories[i] + '/' +
                        img_name.replace('unalt_', 'gamma12_'), gamma12_img)
        print("Finished augmenting training images taken by " + categories[i] + ".")


def extract_patches(path, max_patches, patch_size):
    """
    Extract a patch of images of the same `patch_size` from the original image.
    Output is a 4D-Numpy array with shape (max_patches, *patch_size, num_channels=3).
    """
    img = cv2.imread(path)
    img = np.expand_dims(img, axis=0)
    patch_extractor = PatchExtractor(max_patches=max_patches, patch_size=patch_size)
    patch_extractor.fit(img)
    return patch_extractor.transform(img)


def save_patches(path):
    """
    Loop through every processed image in every phone category to extract patch of images (4D-Numpy array)
    then save it to the directory specified in `path` argument.
    """
    for i in range(len(categories)):
        print('Started extracting patches of processed images taken by ' + categories[i] + '.')
        extracted_path = path + categories[i] + '/'
        for img_name in sorted(os.listdir(extracted_path)):
            patches = extract_patches(path=extracted_path + img_name,
                                      max_patches=20,
                                      patch_size=(299, 299))
            for patch_index, patch in enumerate(patches):
                str_patch_index = str(patch_index)
                cv2.imwrite(extracted_path + 'p' + str_patch_index + img_name, patch)
        print('Finished extracting patches of processed images taken by ' + categories[i] + '.')


def train_test_split_post_augmenting(path=processed_imgs_dir, 
                                     categories=categories,
                                     test_size=0.01):
    """
    Split augmented images into train/ validation sets.
    """
    splitting = dict()
    for i in range(len(categories)):
        splitting[categories[i]] = {}
        all_imgs = sorted(os.listdir(path + categories[i] + '/'))
        compressed70_imgs = [img for img in all_imgs if 'compressed70_' in img]
        compressed90_imgs = [img for img in all_imgs if 'compressed90_' in img]
        resized05_imgs = [img for img in all_imgs if 'resized05_' in img]
        resized08_imgs = [img for img in all_imgs if 'resized08_' in img]
        resized15_imgs = [img for img in all_imgs if 'resized15_' in img]
        resized20_imgs = [img for img in all_imgs if 'resized20_' in img]
        gamma08_imgs = [img for img in all_imgs if 'gamma08_' in img]
        gamma12_imgs = [img for img in all_imgs if 'gamma12_' in img]
        unalt_imgs = [img for img in all_imgs if 'unalt_' in img]
        valid_imgs = np.random.choice(compressed70_imgs, size=int(len(compressed70_imgs)*test_size),
                                      replace=False).tolist()
        valid_imgs += np.random.choice(compressed90_imgs, size=int(len(compressed90_imgs)*test_size),
                                       replace=False).tolist()
        valid_imgs += np.random.choice(resized05_imgs, size=int(len(resized05_imgs)*test_size),
                                       replace=False).tolist()
        valid_imgs += np.random.choice(resized08_imgs, size=int(len(resized08_imgs)*test_size),
                                       replace=False).tolist()
        valid_imgs += np.random.choice(resized15_imgs, size=int(len(resized15_imgs)*test_size),
                                       replace=False).tolist()
        valid_imgs += np.random.choice(resized20_imgs, size=int(len(resized20_imgs)*test_size),
                                       replace=False).tolist()
        valid_imgs += np.random.choice(gamma08_imgs, size=int(len(gamma08_imgs)*test_size),
                                       replace=False).tolist()
        valid_imgs += np.random.choice(gamma12_imgs, size=int(len(gamma12_imgs)*test_size),
                                       replace=False).tolist()
        valid_imgs += np.random.choice(unalt_imgs, size=int(len(unalt_imgs)*test_size),
                                       replace=False).tolist()
        splitting[categories[i]]['validation'] = valid_imgs
        splitting[categories[i]]['train'] = [img for img in all_imgs if img not in valid_imgs]
    return splitting


def center_crop(img, center_crop_size):
    """
    Crop the 299x299 centered block in every image.
    """
    centerw, centerh = img.shape[0] // 2, img.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped_img = img[centerw - halfw:centerw + halfw, 
                      centerh - halfh:centerh + halfh, :]
    return cropped_img


def write_imgs_to_train_and_validation_dirs(input_path=processed_imgs_dir,
                                            output_path1=train_dir,
                                            output_path2=validation_dir):
    """
    Randomly split patched processed images into training and validation sets then copy them to 
    the relevant directories.
    """
    splitting = train_test_split_post_augmenting()
    for i in range(len(categories)):
        for img_name in splitting[categories[i]]["train"]:
            srcfile = input_path + categories[i] + '/' + img_name
            dstroot = output_path1 + categories[i] + '/'
            copy(srcfile, dstroot)
        print("Finished writing augmented images taken by " +
              categories[i] + " to the train directory.")
        for img_name in splitting[categories[i]]["validation"]:
            srcfile = input_path + categories[i] + '/' + img_name
            dstroot = output_path2 + categories[i] + '/'
            copy(srcfile, dstroot)
        print("Finished writing augmented images taken by " +
              categories[i] + " to the validation directory.")
    
if __name__ == "__main__":
    augment_training_img()
    extract_patches(path=processed_imgs_dir, max_patches=20, patch_size=(299, 299))
    save_patches(path=processed_imgs_dir)
    write_imgs_to_train_and_validation_dirs()