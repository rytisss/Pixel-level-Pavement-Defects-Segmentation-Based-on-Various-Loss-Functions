import os
import json
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from models.autoencoder import AutoEncoder4
from models.losses import Loss, AlphaScheduler
from models.data_loader import data_generator
from utilities import gather_image_from_dir

# Data
image_width = 480
image_height = 320
image_channels = 1

# train
train_images_dir = r'D:\pavement defect data\CrackForestdatasets\datasets\Set_0\Train\AUGM\Images/'
train_labels_dir = r'D:\pavement defect data\CrackForestdatasets\datasets\Set_0\Train\AUGM\Labels/'
# test
test_images_dir = r'D:\pavement defect data\CrackForestdatasets\datasets\Set_0\Test\Images/'
test_labels_dir = r'D:\pavement defect data\CrackForestdatasets\datasets\Set_0\Test\Labels/'

# Directory for weight saving (creates if it does not exist)
weights_output_dir = 'weights_output/'
weights_output_name = 'UNet4'
# batch size. How many samples you want to feed in one iteration?
batch_size = 4
# number_of_epoch. How many epochs you want to train?
number_of_epoch = 50
# learning rate
lr = 0.001

# predefined all the loss functions for training. Separate training will be conducted to each of these functions
loss_dict = {
    'boundary': Loss.SURFACEnDice,
    'cross_entropy': Loss.CROSSENTROPY,
    'dice': Loss.DICE,
    'weighted_cross_entropy': Loss.WEIGHTEDCROSSENTROPY,
    'weighted60_cross_entropy': Loss.WEIGHTED60CROSSENTROPY,
    'weighted70_cross_entropy': Loss.WEIGHTED70CROSSENTROPY,
    'cross_entropy_50_dice_50': Loss.CROSSENTROPY50DICE50,
    'cross_entropy_25_dice_75': Loss.CROSSENTROPY25DICE75,
    'cross_entropy_75_dice_25': Loss.CROSSENTROPY75DICE25,
    'weighted_cross_entropy_50_dice_50': Loss.WEIGHTEDCROSSENTROPY50DICE50,
    'weighted_cross_entropy_25_dice_75': Loss.WEIGHTEDCROSSENTROPY25DICE75,
    'weighted_cross_entropy_75_dice_25': Loss.WEIGHTEDCROSSENTROPY75DICE25
}


class CustomSaver(tf.keras.callbacks.Callback):
    def __init__(self):
        self.best_val_score = 0.0

    def on_epoch_end(self, epoch, logs=None):
        # also save if validation error is smallest
        if 'val_dice_eval' in logs.keys():
            val_score = logs['val_dice_eval']
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                print('New best weights found!')
                self.model.save(weights_output_dir + 'best_weights.hdf5')
        else:
            print('Key val_dice_eval does not exist!')


def train():
    for loss_name, loss in loss_dict.items():
        tf.keras.backend.clear_session()
        # check how many train and test samples are in the directories
        train_images_count = len(gather_image_from_dir(train_images_dir))
        train_labels_count = len(gather_image_from_dir(train_labels_dir))
        train_samples_count = min(train_images_count, train_labels_count)
        print('Training samples: ' + str(train_samples_count))

        test_images_count = len(gather_image_from_dir(test_images_dir))
        test_labels_count = len(gather_image_from_dir(test_labels_dir))
        test_samples_count = min(test_images_count, test_labels_count)
        print('Testing samples: ' + str(test_samples_count))

        # how many iterations in one epoch? Should cover whole dataset. Divide number of data samples from batch size
        number_of_train_iterations = train_samples_count // batch_size
        number_of_test_iterations = test_samples_count // batch_size

        # Define model
        model = AutoEncoder4(input_size=(image_height, image_width, image_channels), loss_function=loss)

        model.summary()

        # Define data generator that will take images from directory
        train_data_generator = data_generator(batch_size,
                                              image_folder=train_images_dir,
                                              label_folder=train_labels_dir,
                                              target_size=(image_height, image_width),
                                              image_color_mode='grayscale')

        test_data_generator = data_generator(batch_size,
                                             image_folder=test_images_dir,
                                             label_folder=test_labels_dir,
                                             target_size=(image_height, image_width),
                                             image_color_mode='grayscale')

        # create weights output directory
        if not os.path.exists(weights_output_dir):
            print('Output directory doesnt exist!\n')
            print('It will be created!\n')
            os.makedirs(weights_output_dir)

        # Custom saving for the best-performing weights
        saver = CustomSaver()

        if loss == Loss.SURFACEnDice:
            scheduler = AlphaScheduler()
            train_history = model.fit(train_data_generator,
                                      steps_per_epoch=number_of_train_iterations,
                                      epochs=number_of_epoch,
                                      validation_data=test_data_generator,
                                      validation_steps=number_of_test_iterations,
                                      callbacks=[saver, scheduler],
                                      shuffle=True)
        else:
            train_history = model.fit(train_data_generator,
                                      steps_per_epoch=number_of_train_iterations,
                                      epochs=number_of_epoch,
                                      validation_data=test_data_generator,
                                      validation_steps=number_of_test_iterations,
                                      callbacks=[saver],
                                      shuffle=True)

        output_file = open(loss_name + '.json', 'w')
        json.dump(train_history.history, output_file)
        output_file.close()


if __name__ == "__main__":
    train()
