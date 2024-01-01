from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, UpSampling3D, Input
from tensorflow.keras.models import Model

def EncoderDecoder_Model(input_shape):
    in_layer = Input(input_shape)
    t = Conv3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(in_layer)
    t = Conv3D(64, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(t)
    t = MaxPooling3D((2, 2, 2))(t)
    t = Conv3D(128, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(t)
    t = Conv3D(128, kernel_size=(3, 3, 3), padding = 'same', activation = 'relu')(t)
    t = MaxPooling3D((2, 2, 2))(t)

    t = Conv3DTranspose(128, kernel_size=(3, 3, 3), padding='same', activation='relu')(t)
    t = Conv3DTranspose(128, kernel_size=(3, 3, 3), padding='same', activation='relu')(t)
    t = UpSampling3D((2, 2, 2))(t)
    t = Conv3DTranspose(64, kernel_size=(3, 3, 3), padding='same', activation='relu')(t)
    t = Conv3DTranspose(64, kernel_size=(3, 3, 3), padding='same', activation='relu')(t)
    t = UpSampling3D((2, 2, 2))(t)

    t = Conv3D(1, kernel_size=(3, 3, 3), padding='same', activation='relu')(t)

    return Model(in_layer, t, name='EncoderDecoder')

    





