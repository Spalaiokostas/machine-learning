import tensorflow as tf
import os
import pandas as pd
import numpy as np

def _read_files(csv_filename, images_filepath):

    # assert that files exist
    csv_filepath = os.path.join(os.getcwd(), csv_filename)
    if not os.path.isfile(csv_filepath):
        print("Could not find the csv file with the images and their labels")
        exit(-1)

    if not os.path.isdir(os.path.join(os.getcwd(), images_filepath)):
        print("Could not find the directory containing the images")
        exit(-1)

    # use pandas to read the csv
    dataframe = pd.read_csv(csv_filepath)

    # concatenate image file name with the image directory
    dataframe['ID'] = dataframe['ID'].apply(lambda x: os.path.join(images_filepath, x))

    return dataframe

def train_and_eval_sets(csv_filename, images_filepath):

    dataframe = _read_files(csv_filename,images_filepath)

    # change labels from string to integer representation (we later one-hot them)
    dataframe = dataframe.replace(['YOUNG', 'MIDDLE', 'OLD'], [0, 1, 2])

    # will be used to split between train and validate
    mask = np.random.rand(len(dataframe)) < 0.8
    train = dataframe[mask]
    validate = dataframe[~mask]

    return train, validate


def prediction_set(csv_filename, images_filepath):
    return _read_files(csv_filename, images_filepath)


def __preprocess_img(image_name):

    # load and preprocess image
    image = tf.image.decode_jpeg(tf.read_file(image_name), channels=3)
    resize_image = tf.image.resize_images(image, [64, 64])

    # convert to grayscale and normalize
    image = tf.image.rgb_to_grayscale(resize_image)
    image = tf.image.per_image_standardization(image)

    return image

def _parse_image_and_label(img_name, label):

    # one-hot labels
    label = tf.one_hot(label, depth=3)

    image = __preprocess_img(img_name)

    return image, label


# input functions used by estimator
def input_fn(dataframe, epoch, suffle, batch_size, buffer_size = 500):

     # for images
    img_paths = tf.convert_to_tensor(
        dataframe["ID"].tolist(),
        dtype=tf.string)

    # for labels (exist only in training mode
    labels = tf.convert_to_tensor(
        dataframe["Class"].tolist(),
        dtype=tf.uint8)


    inputs = (img_paths, labels)

    # create Dataset from tensors
    data_set = tf.data.Dataset.from_tensor_slices(inputs)

    if suffle:
        data_set = data_set.shuffle(buffer_size=buffer_size)
    data_set = data_set.map(_parse_image_and_label, num_parallel_calls=4)

    # if num_epochs is None then the element of the dataset are repeated indefinitely
    data_set = data_set.repeat(epoch)
    data_set = data_set.batch(batch_size)

    return data_set.make_one_shot_iterator().get_next()


def predict_input_fn(dataframe, batch_size):
    # for images
    img_paths = tf.convert_to_tensor(
        dataframe["ID"].tolist(),
        dtype=tf.string)

    # create Dataset from tensors
    data_set = tf.data.Dataset.from_tensor_slices(img_paths)

    data_set = data_set.map(__preprocess_img, num_parallel_calls=4)

    # if num_epochs is None then the element of the dataset are repeated indefinitely
    data_set = data_set.repeat(1)
    data_set = data_set.batch(batch_size)

    return data_set.make_one_shot_iterator().get_next()

def store_predicted_labels(dataframe, predictions, csv_filename):
    # wtite predictions to test CSV file
    labels = ['YOUNG', 'MIDDLE', 'OLD']

    predicted_labels = []
    for predict in predictions:
        predicted_labels.append(labels[predict["classes"]])

    dataframe['Class'] = pd.Series(predicted_labels)

    csv_filepath = os.path.join(os.getcwd(), csv_filename)
    dataframe.to_csv(csv_filepath, sep='\t', encoding='utf-8')