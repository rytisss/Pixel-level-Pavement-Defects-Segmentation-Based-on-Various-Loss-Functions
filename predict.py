import cv2

from models.autoencoder import AutoEncoder4
from models.losses import Loss
from processing import tensor_to_image, image_to_tensor
from utilities import gather_image_from_dir

import tensorflow as tf
# Weights path
weight_path = 'weights_output/best_weights.hdf5' # provide path to weights '*.hdf5'
# Test images directory
test_images = r'D:\pavement defect data\CrackForestdatasets\datasets\Set_0\Test\Images/'

image_width = 480
image_height = 320
image_channels = 1

def predict():
    # Define model
    model = AutoEncoder4(input_size=(image_height, image_width, image_channels),
                         loss_function=Loss.CROSSENTROPY,
                         pretrained_weights=weight_path)

    image_paths = gather_image_from_dir(test_images)

    # Load and predict on all images from directory
    for image_path in image_paths:
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # preprocess
        norm_image = image_to_tensor(image)
        # predict
        prediction = model.predict(norm_image)
        # make image uint8
        prediction_image = tensor_to_image(prediction)

        # Do you want to visualize image?
        show_image = True
        if show_image:
            cv2.imshow("image", image)
            cv2.imshow("prediction", prediction_image)
            cv2.waitKey(1000)


if __name__ == '__main__':
    predict()
