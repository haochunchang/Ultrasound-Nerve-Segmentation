from keras.layers import Input, concatenate, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam, SGD
from keras.models import Model, model_from_json, Sequential
from keras import backend as K
from os.path import join

def load_model(model_path):
    with open(join(model_path, 'model.json'), 'r') as f:
        model = model_from_json(f.read())
        
    model.load_weights(join(model_path, 'weights.h5'))
    return model

#====================
# Custom Score Metric
#====================
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def pixel_wise_entropy_loss(y_true, y_pred):
    pass
    
#===============
# Model Class
#===============
class ModelBuilder(object):
    """
    Class for building NN models
    input_shape, output_shape: tuple (image_height (row), image_width (cols))
    """
    def __init__(self, input_shape, output_shape):
        self.inp_shape = input_shape
        self.out_shape = output_shape
        
    
    def build_model(self, kind='Unet'):
        """
        Returns: keras compiled model object
        """
        try:
            model = getattr(self, '_build_{}'.format(kind))()
            return model
        except:
            print(kind)
            raise NotImplementedError

    
    def _build_Unet(self):
        
        inputs = Input((self.inp_shape[0], self.inp_shape[1], 1))
        
        conv1 = Conv2D(32, (3, 3), activation='elu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(32, (3, 3), activation='elu', padding='same')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='elu', padding='same')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='elu', padding='same')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='elu', padding='same')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(256, (3, 3), activation='elu', padding='same')(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='elu', padding='same')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(512, (3, 3), activation='elu', padding='same')(conv5)
        conv5 = BatchNormalization()(conv5)
        
        # Start Unet Upsampling
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5),
                           conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='elu', padding='same')(up6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(256, (3, 3), activation='elu', padding='same')(conv6)
        conv6 = BatchNormalization()(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6),
                           conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='elu', padding='same')(up7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv7)
        conv7 = BatchNormalization()(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7),
                           conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='elu', padding='same')(up8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv8)
        conv8 = BatchNormalization()(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8),
                           conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='elu', padding='same')(up9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(32, (3, 3), activation='elu', padding='same')(conv9)
        conv9 = BatchNormalization()(conv9)

        segment = Conv2D(1, (1, 1), activation='sigmoid', name="Segment")(conv9)

        # Auxilliary layers for predicting Nerve presence (0/1)
        conv_1x1 = Conv2D(256, (1, 1), activation='elu')(conv5)
        flat = GlobalAveragePooling2D()(conv_1x1)
        #fc1 = Dense(256, activation='elu')(flat)
        clf = Dense(2, activation='sigmoid', name="Presence")(flat)
        
        model = Model(inputs=[inputs], outputs=[segment, clf])
        losses = {
            "Segment": dice_coef_loss,
            "Presence": "binary_crossentropy"
        }
        lossWeights = {"Segment": 1.0, "Presence": 1.0}
        model.compile(optimizer=Adam(lr=2e-5), loss=losses, loss_weights=lossWeights,
                      metrics=[dice_coef, "accuracy"])

        return model

    
    def _build_SimpleCNN(self):
        
        model = Sequential()
        model.add(Conv2D(4, (4, 4), padding='same', kernel_initializer='he_normal', activation='relu',
                                input_shape=(self.inp_shape[0], self.inp_shape[1], 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(8, (4, 4), padding='same', kernel_initializer='he_normal', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.6))

        model.add(Flatten())
        model.add(Dense(2, activation='softmax'))

        #sgd = SGD(lr=1.00032e-3, decay=.8e-6, momentum=0.901, nesterov=True)
        model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
