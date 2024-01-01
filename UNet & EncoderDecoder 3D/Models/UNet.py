from tensorflow.keras.layers import ConvLSTM2D, Bidirectional, BatchNormalization, Conv3D, Cropping3D, ZeroPadding3D, Activation, Input
from tensorflow.keras.layers import MaxPooling3D, UpSampling3D, Conv3DTranspose, concatenate
from tensorflow.keras.models import Model

def UNet_Model(input_shape):
   in_layer = Input(input_shape)
   bn = BatchNormalization()(in_layer)
   cn1 = Conv3D(8, 
               kernel_size = (1, 5, 5), 
               padding = 'same',
               activation = 'relu')(bn)
   cn2 = Conv3D(8, 
               kernel_size = (3, 3, 3),
               padding = 'same',
               activation = 'linear')(cn1)
   bn2 = Activation('relu')(BatchNormalization()(cn2))

   dn1 = MaxPooling3D((2, 2, 2))(bn2)
   cn3 = Conv3D(16, 
               kernel_size = (3, 3, 3),
               padding = 'same',
               activation = 'linear')(dn1)
   bn3 = Activation('relu')(BatchNormalization()(cn3))

   dn2 = MaxPooling3D((1, 2, 2))(bn3)
   cn4 = Conv3D(32, 
               kernel_size = (3, 3, 3),
               padding = 'same',
               activation = 'linear')(dn2)
   bn4 = Activation('relu')(BatchNormalization()(cn4))

   up1 = Conv3DTranspose(16, 
                        kernel_size = (3, 3, 3),
                        strides = (1, 2, 2),
                        padding = 'same')(bn4)

   cat1 = concatenate([up1, bn3])

   up2 = Conv3DTranspose(8, 
                        kernel_size = (3, 3, 3),
                        strides = (2, 2, 2),
                        padding = 'same')(cat1)

   pre_out = concatenate([up2, bn2])

   pre_out = Conv3D(1, 
               kernel_size = (1, 1, 1), 
               padding = 'same',
               activation = 'sigmoid')(pre_out)

   pre_out = Cropping3D((1, 2, 2))(pre_out) # avoid skewing boundaries
   out = ZeroPadding3D((1, 2, 2))(pre_out)
   return Model(inputs = [in_layer], outputs = [out])

