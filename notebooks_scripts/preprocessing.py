# -*- coding: utf-8 -*-
"""
image preprocessing functions based on the OpenCV library 
"""
import numpy as np
import os
import cv2
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


def random_jpeg_compress(path):
    quality_factor = np.random.choice([70, 90])
    img = cv2.imread(path)
    if quality_factor == 70:
        result, encoded_img = cv2.imencode(
            '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        decoded_img = cv2.imdecode(encoded_img, 1)
    else:
        result, encoded_img = cv2.imencode(
            '.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        decoded_img = cv2.imdecode(encoded_img, 1)
    return decoded_img


# resize via bicubic interpolation
def random_resize(path):
    resizing_factor = np.random.choice([0.5, 0.8, 1.5, 2.0])
    img = cv2.imread(path)
    resized_img = cv2.resize(img, None,
                             fx=resizing_factor, fy=resizing_factor,
                             interpolation=cv2.INTER_CUBIC)
    return resized_img


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


def random_gamma_correction(path):
    img = cv2.imread(path)
    gamma = np.random.choice([0.8, 1.2])
    inv_gamma = 1.0 / gamma
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    lookup_table = np.array([((i / 255.0) ** inv_gamma) * 255
                             for i in np.arange(0, 256)]).astype("uint8")
    corrected_img = cv2.LUT(img, lookup_table)
    return corrected_img


# randomly choose one of the operations
def alter_training_img(path):
    operation = np.random.choice(["compression", "resize", "gamma_correction"])
    if operation == "compression":
        img = random_jpeg_compress(path)
    elif operation == "resize":
        img = random_resize(path)
    else:
        img = random_gamma_correction(path)
    return img


# augment training images with all image alterations
def augment_training_img(input_path=raw_imgs_dir, output_path=processed_imgs_dir):
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

        
# in the test set, half the number of images was altered by one of the transformations
# randomly sample 137 images from each category where 137 = (2750/2)//10
def random_sample_preprocessing(path=raw_imgs_dir, categories=categories,
                                altered_size_per_category=137, random_state=10):
    np.random.seed(random_state)  # for reproducibility
    partition = dict()

    for i in range(len(categories)):  # looping through each category folder
        partition[categories[i]] = {}
        partition[categories[i]]["altered"] = np.random.choice(
            sorted(os.listdir(path + categories[i])),
            size=altered_size_per_category, replace=False).tolist()
        partition[categories[i]]["non-altered"] = [
            img_name for img_name in sorted(os.listdir(path + categories[i])) 
            if img_name not in partition[categories[i]]["altered"]]
        
    return partition


#  apply transformation on the original image and then write to new directory
def transform_and_write(input_path=raw_imgs_dir, output_path=processed_imgs_dir):
    training_partition = random_sample_preprocessing()
    for i in range(len(categories)):
        print("Started pre-processing images taken by " + categories[i] + ".")
        for img_name in training_partition[categories[i]]["altered"]:
            altered_img = alter_training_img(
                input_path + categories[i] + '/' + img_name)
            cv2.imwrite(output_path +
                        categories[i] + '/' + 'alt_' + img_name, altered_img)
        for img_name in training_partition[categories[i]]["non-altered"]:
            copied_img = cv2.imread(
                input_path + categories[i] + '/' + img_name)
            cv2.imwrite(output_path +
                        categories[i] + '/' + 'unalt_' + img_name, copied_img)
        print("Finished pre-processing images taken by " + categories[i] + ".")

# split processed images into train/ validation sets
def train_test_split_post_random_preprocessing(path=processed_imgs_dir, categories=categories,
                                               alt_test_size_per_category=28,
                                               unalt_test_size_per_category=27,
                                               random_state=11):
    np.random.seed(random_state)
    splitting = dict()

    for i in range(len(categories)):
        splitting[categories[i]] = {}
        unalt_imgs = [img_name for img_name in sorted(
            os.listdir(path + categories[i])) if img_name.startswith('unalt_')]
        alt_imgs = [img_name for img_name in sorted(os.listdir(
            path + categories[i])) if img_name not in unalt_imgs]
        val_unalt_imgs = np.random.choice(unalt_imgs,
                                          size=unalt_test_size_per_category,
                                          replace=False).tolist()
        val_alt_imgs = np.random.choice(alt_imgs,
                                        size=alt_test_size_per_category,
                                        replace=False).tolist()
        splitting[categories[i]]["validation"] = val_unalt_imgs + val_alt_imgs
        splitting[categories[i]]["train"] = [
            img_name for img_name in sorted(os.listdir(path + categories[i])) 
            if img_name not in splitting[categories[i]]["validation"]]
        assert len(splitting[categories[i]]["train"]) == 220

    return splitting


# split augmented images into train/ validation sets
def train_test_split_post_augmenting(path=processed_imgs_dir, 
                                     categories=categories,
                                     test_size=0.1):
    splitting = dict()
    for i in range(len(categories)):
        splitting[categories[i]] = {}
        all_imgs = sorted(os.listdir(path + categories[i] + '/'))
        compressed70_imgs = [img for img in all_imgs if img.startswith('compressed70_')]
        compressed90_imgs = [img for img in all_imgs if img.startswith('compressed90_')]
        resized05_imgs = [img for img in all_imgs if img.startswith('resized05_')]
        resized08_imgs = [img for img in all_imgs if img.startswith('resized08_')]
        resized15_imgs = [img for img in all_imgs if img.startswith('resized15_')]
        resized20_imgs = [img for img in all_imgs if img.startswith('resized20_')]
        gamma08_imgs = [img for img in all_imgs if img.startswith('gamma08_')]
        gamma12_imgs = [img for img in all_imgs if img.startswith('gamma12_')]
        unalt_imgs = [img for img in all_imgs if img.startswith('unalt_')]
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


# crop the 512x512 centered block in each image and save to training data directory
def center_crop(img, center_crop_size):
    centerw, centerh = img.shape[0] // 2, img.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped_img = img[centerw - halfw:centerw + halfw, 
                      centerh - halfh:centerh + halfh, :]
    return cropped_img

def crop_and_write(input_path=processed_imgs_dir,
                   output_path1=train_dir,
                   output_path2=validation_dir):
    splitting = train_test_split_post_augmenting()
    for i in range(len(categories)):
        print("Started cropping processed images taken by " +
              categories[i] + ".")
        for img_name in splitting[categories[i]]["train"]:
            train_img = cv2.imread(input_path +
                                   categories[i] + '/' + img_name)
            cropped_train_img = center_crop(
                train_img, center_crop_size=(512, 512))
            cv2.imwrite(output_path1 +
                        categories[i] + '/' + img_name, cropped_train_img)
        for img_name in splitting[categories[i]]["validation"]:
            val_img = cv2.imread(input_path +
                                 categories[i] + '/' + img_name)
            cropped_val_img = center_crop(val_img, center_crop_size=(512, 512))
            cv2.imwrite(output_path2 +
                        categories[i] + '/' + img_name, cropped_val_img)
        print("Finished cropping processed images taken by " +
              categories[i] + ".")
    
if __name__ == "__main__":
    augment_training_img()
    crop_and_write()