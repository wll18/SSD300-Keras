import keras.backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import tensorflow as tf
from keras.layers import Activation, AtrousConvolution2D, Convolution2D, Dense, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D, concatenate,  Model, Reshape, Conv2D, ZeroPadding2D,MaxPooling2D, MaxPool2D

from keras.applications.imagenet_utils import _obtain_input_shape


class SSD(object):
	""" """
	def __init__(self, input_shape = (300,300,3), number_classes=21):
		self.number_priors = 6
		self.voc_classes_genre = {'Avion' : 'un', 'Vélo' : 'un', 'Oiseau' : 'un', 'Bateau' : 'un', 'Bouteille' : 'une','Bus' : 'un', 'Voiture' : 'une', 'Chat' : 'un', 'Chaise' : 'une', 'Vache' : 'une', 'Table' : 'une',
               'Chien' : 'un', 'Cheval' : 'un','Moto' : 'une', 'Personne' : 'une', 'Plante' : 'une',
               'Mouton' : 'un', 'Canapé' : 'un', 'Train' : 'un', 'Télévision' : 'une'}
        	self.genre = ['une', 'un']
		self.NUM_CLASSES = len(voc_classes) + 1
		self.input_shape=(300, 300, 3)


	def call(self, input_shape = (300,300,3), num_classes=21):

		K.set_image_dim_ordering('tf')   

		input_shape = _obtain_input_shape(input_shape,default_size=300,min_size=48,data_format=K.image_data_format(),include_top=True)
		input_tensor = input_tensor = Input(shape=self.input_shape)
		self.img_size = (input_shape[1], input_shape[0])

		#### Architecture de base pour l'extraction des caractéristiques basé sur VGG16 ####

		x, CONV_4, FC_7, CONV_6, CONV_7,CONV_8, POOL_6 = self.VGG16_Base(input_tensor)

		#### positions, probabilités et coordonées des box ####

		location_conv_4, confidence_conv_4, priorbox_conv_4 = self.PredictionLayer(Normalize(20)(CONV_4))
		location_fulc_7, confidence_fulc_7, priorbox_fulc_7 = self.PredictionLayer(FC_7)
		location_conv_6, confidence_conv_6, priorbox_conv_6 = self.PredictionLayer(CONV_6)
		location_conv_7, confidence_conv_7, priorbox_conv_7 = self.PredictionLayer(CONV_7)
		location_conv_8, confidence_conv_8, priorbox_conv_8 = self.PredictionLayer(CONV_8)

		location_pool_6 = Dense(num_priors * 4)(x)
		confidence_pool_6 = Dense(num_priors * num_classes)(POOL_6)

		priorbox_pool_6_temp = PriorBox(self.img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2])
		priorbox_pool_6_shape = Reshape((1, 1, 256))(POOL_6)
		priorbox_pool_6 = priorbox(priorbox_pool_6_shape)

		#### concaténation des positions, des probabilités, et des coordonées aux différents niveaux ####

		locations = concatenate([location_conv_4,location_fulc_7,location_conv_6,location_conv_7,location_conv_8,location_pool_6],axis=1)
		confidences = concatenate([confidence_conv_4,confidence_fulc_7,confidence_conv_6,confidence_conv_7,confidence_conv_8,confidence_pool_6], axis=1)
		boxes = concatenate([priorbox_conv_4,priorbox_fulc_7,priorbox_conv_6,priorbox_conv_7,priorbox_conv_8,priorbox_pool_6],axis=1)

		if hasattr(locations, '_keras_shape'):
			num_boxes = locations._keras_shape[-1] // 4
		elif hasattr(locations, 'int_shape'):
			num_boxes = K.int_shape(locations)[-1] // 4

		locations = Reshape((num_boxes, 4))(locations)
		confidences = Reshape((num_boxes, num_classes))(confidences)
		confidences = Activation('softmax')(confidences)

		predictions = concatenate([locations,confidences,boxes],axis=2)

		model = Model(input_tensor, predictions)

		return model

	def PredictionLayer(self, conv_layer):

	    conv_mbox_loc = Convolution2D(self.number_priors * 4, 3, 3, border_mode='same')(conv_layer)
	    conv_mbox_loc_flat = Flatten()(conv_mbox_loc)

	    conv_mbox_conf = Convolution2D(self.number_priors * num_classes, 3, 3, border_mode='same')(conv_layer)
	    conv_mbox_conf_flat = Flatten()(conv_mbox_conf)

	    conv_mbox_priorbox = PriorBox(self.img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3],variances=[0.1, 0.1, 0.2, 0.2])(conv_layer)

	    return conv_mbox_loc, conv_mbox_conf, conv_mbox_priorbox

	def VGG16_Base(self,input_tensor):

		x = Convolution2D(64, 3, 3,activation='relu',border_mode='same')(input_tensor)
		x = Convolution2D(64, 3, 3,activation='relu',border_mode='same')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(x)

		x = Convolution2D(128, 3, 3,activation='relu',border_mode='same')(x)
		x = Convolution2D(128, 3, 3,activation='relu',border_mode='same')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(x)

		x = Convolution2D(256, 3, 3,activation='relu',border_mode='same')(x)
		x = Convolution2D(256, 3, 3,activation='relu',border_mode='same')(x)
		x = Convolution2D(256, 3, 3,activation='relu',border_mode='same')(x)
		x = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same')(x)

		x = Convolution2D(512, 3, 3,activation='relu',border_mode='same')(x)
		x = Convolution2D(512, 3, 3,activation='relu',border_mode='same')(x)
		x = Convolution2D(512, 3, 3,activation='relu',border_mode='same')(x)

		CONV_4 = x

		x = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',name='MaxPool_4')(x)

		x = Convolution2D(512, 3, 3,activation='relu',border_mode='same')(net['pool4'])
		x = Convolution2D(512, 3, 3,activation='relu',border_mode='same')(x)
		x = Convolution2D(512, 3, 3,activation='relu',border_mode='same')(x)
		x = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(x)

		x = AtrousConvolution2D(1024, 3, 3, atrous_rate=(6, 6),activation='relu', border_mode='same')(x)
		x = Convolution2D(1024, 1, 1, activation='relu',border_mode='same')(x)

		FC_7 = x

		x = Convolution2D(256, 1, 1, activation='relu',border_mode='same')(x)
		x = Convolution2D(512, 3, 3, subsample=(2, 2),activation='relu', border_mode='same')(x)

		CONV_6 = x

		x = Convolution2D(128, 1, 1, activation='relu',border_mode='same')(x)
		x = ZeroPadding2D()(x)
		x = Convolution2D(256, 3, 3, subsample=(2, 2),activation='relu', border_mode='valid')(x)

		CONV_7 = x

		x = Convolution2D(128, 1, 1, activation='relu',border_mode='same')(x)
		x = Convolution2D(256, 3, 3, subsample=(2, 2),activation='relu', border_mode='same')(x)

		CONV_8 = x

		x = GlobalAveragePooling2D(name='GaPooling_6')(x)

		POOL_6 = x

		return x, CONV_4, FC_7, CONV_6, CONV_7, CONV_8, POOL_6

	def BuildModel(self, weight_path):
		model = self.SSD300(self.input_shape, num_classes=self.NUM_CLASSES)
		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

		model.compile(optimizer=sgd, loss='categorical_crossentropy')
		model.summary()
		model.load_weights(weight_path, by_name=True)

		return model
