import math

import numpy as np
import tensorflow as tf

"""from tensorflow.keras.layers import LeakyReLU, Activation, Conv2DTranspose, Conv2D, GlobalAveragePooling2D, \
    BatchNormalization, Dense, AveragePooling2D, concatenate, UpSampling2D, Add, MaxPooling2D, Input"""
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Activation, Dense, BatchNormalization, LeakyReLU, \
    AveragePooling2D, UpSampling2D, concatenate, Add, MaxPooling2D, Conv2DTranspose, Input
# from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.losses import Loss
from models.losses import dice_loss, dice_score, dice_eval, binary_crossentropy, Active_Contour_Loss, surface_loss, FocalLoss, \
    weighted_bce_loss, adjusted_weighted_bce_loss, cross_and_dice_loss, \
    weighted_cross_and_dice_loss, cross_and_dice_loss_multiclass, surficenDiceLoss


def CompileModel(model, lossFunction, num_class=2, learning_rate=1e-3):
    if num_class == 2:
        if lossFunction == Loss.DICE:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=dice_loss, metrics=[dice_eval])
        elif lossFunction == Loss.CROSSENTROPY:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=binary_crossentropy, metrics=[dice_eval])
        elif lossFunction == Loss.ACTIVECONTOURS:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Active_Contour_Loss, metrics=[dice_eval])
        elif lossFunction == Loss.SURFACEnDice:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=surficenDiceLoss, metrics=[dice_eval])
        elif lossFunction == Loss.FOCALLOSS:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=FocalLoss, metrics=[dice_eval])
        elif lossFunction == Loss.WEIGHTEDCROSSENTROPY:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=weighted_bce_loss, metrics=[dice_eval])
        elif lossFunction == Loss.WEIGHTED60CROSSENTROPY:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=adjusted_weighted_bce_loss(0.6), metrics=[dice_eval])
        elif lossFunction == Loss.WEIGHTED70CROSSENTROPY:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=adjusted_weighted_bce_loss(0.7), metrics=[dice_eval])
        elif lossFunction == Loss.CROSSENTROPY50DICE50:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cross_and_dice_loss(0.5, 0.5), metrics=[dice_eval])
        elif lossFunction == Loss.CROSSENTROPY25DICE75:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cross_and_dice_loss(0.25, 0.75), metrics=[dice_eval])
        elif lossFunction == Loss.CROSSENTROPY75DICE25:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cross_and_dice_loss(0.75, 0.25), metrics=[dice_eval])
        elif lossFunction == Loss.WEIGHTEDCROSSENTROPY50DICE50:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=weighted_cross_and_dice_loss(0.5, 0.5),
                          metrics=[dice_eval])
        elif lossFunction == Loss.WEIGHTEDCROSSENTROPY25DICE75:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=weighted_cross_and_dice_loss(0.25, 0.75),
                          metrics=[dice_eval])
        elif lossFunction == Loss.WEIGHTEDCROSSENTROPY75DICE25:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=weighted_cross_and_dice_loss(0.75, 0.25),
                          metrics=[dice_eval])
    else:
        if lossFunction == Loss.CROSSENTROPY50DICE50:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cross_and_dice_loss_multiclass(0.5, 0.5),
                          metrics=[dice_eval])
        elif lossFunction == Loss.CROSSENTROPY25DICE75:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cross_and_dice_loss_multiclass(0.25, 0.75),
                          metrics=[dice_eval])
        elif lossFunction == Loss.CROSSENTROPY75DICE25:
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss=cross_and_dice_loss_multiclass(0.75, 0.25),
                          metrics=[dice_eval])
    return model


def EncodingLayer(input,
                  kernels=8,
                  kernel_size=3,
                  stride=1,
                  max_pool=True,
                  max_pool_size=2,
                  batch_norm=True):
    # Double convolution according to U-Net structure
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=stride, padding='same',
                  kernel_initializer='he_normal')(input)
    # Batch-normalization on demand
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    # Max-pool on demand
    if max_pool == True:
        oppositeConnection = conv
        output = MaxPooling2D(pool_size=(max_pool_size, max_pool_size))(conv)
    else:
        oppositeConnection = conv
        output = conv
    # in next step this output needs to be activated
    return oppositeConnection, output


def DecodingLayer(input,
                  skippedInput,
                  upSampleSize=2,
                  kernels=8,
                  kernel_size=3,
                  batch_norm=True):
    conv = Conv2D(kernels, kernel_size=(2, 2), padding='same', kernel_initializer='he_normal')(
        UpSampling2D((upSampleSize, upSampleSize))(input))
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    concatenatedInput = concatenate([conv, skippedInput])
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(concatenatedInput)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(kernels, kernel_size=(kernel_size, kernel_size), strides=1, padding='same',
                  kernel_initializer='he_normal')(conv)
    if batch_norm == True:
        conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    output = conv
    return output

# 4-layer UNet
def AutoEncoder4(pretrained_weights=None,
                 input_size=(320, 480, 1),
                 kernel_size=3,
                 number_of_kernels=32,
                 stride=1,
                 max_pool=True,
                 max_pool_size=2,
                 batch_norm=True,
                 loss_function=Loss.CROSSENTROPY):
    # Input
    inputs = Input(input_size)
    # encoding
    oppositeEnc0, enc0 = EncodingLayer(inputs, number_of_kernels, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc1, enc1 = EncodingLayer(enc0, number_of_kernels * 2, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    oppositeEnc2, enc2 = EncodingLayer(enc1, number_of_kernels * 4, kernel_size, stride, max_pool, max_pool_size,
                                       batch_norm)
    # bottleneck without residual (might be without batch-norm)
    # opposite connection is equal to enc4
    oppositeEnc3, enc3 = EncodingLayer(enc2, number_of_kernels * 8, kernel_size, stride, False, max_pool_size,
                                       batch_norm)
    # decoding
    # Upsample rate needs to be same as down-sampling (pooling)! It will be equal to the stride and max_pool_size product in opposite (encoding layer)
    dec2 = DecodingLayer(enc3, oppositeEnc2, 2, number_of_kernels * 4, kernel_size, batch_norm)
    dec1 = DecodingLayer(dec2, oppositeEnc1, 2, number_of_kernels * 2, kernel_size, batch_norm)
    dec0 = DecodingLayer(dec1, oppositeEnc0, 2, number_of_kernels, kernel_size, batch_norm)

    dec0 = Conv2D(2, kernel_size=(kernel_size, kernel_size), strides=1, padding='same', kernel_initializer='he_normal')(
        dec0)
    if batch_norm:
        dec0 = BatchNormalization()(dec0)
    dec0 = Activation('relu')(dec0)

    outputs = Conv2D(1, (1, 1), padding='same', activation='sigmoid', kernel_initializer='glorot_normal')(dec0)
    model = Model(inputs, outputs)
    # Compile with selected loss function
    model = CompileModel(model, loss_function)
    # Load trained weights if they are passed here
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    #plot_model(model, to_file='AutoEncoder4.png', show_shapes=True, show_layer_names=True)
    return model
