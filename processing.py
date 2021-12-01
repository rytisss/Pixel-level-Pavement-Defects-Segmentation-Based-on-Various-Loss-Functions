import cv2
import numpy as np

def image_to_tensor(image):
    # preprocess
    image_norm = image / 255
    image_norm = np.reshape(image_norm, image_norm.shape + (1,))
    image_norm = np.reshape(image_norm, (1,) + image_norm.shape)
    return image_norm


def tensor_to_image(tensor):
    # normalize to image
    prediction_image_norm = tensor[0, :, :, 0]
    prediction_image = prediction_image_norm * 255
    prediction_image = prediction_image.astype(np.uint8)
    return prediction_image
