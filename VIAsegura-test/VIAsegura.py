import configparser
import json
import tensorflow as tf
import numpy as np
import json

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# for physical_device in physical_devices:
# 	tf.config.experimental.set_memory_growth(physical_device, True)

class Preprocess:
	def __init__(self):
		pass


	def get_image_groups(self, images):
		"""
		Generatethe grouped images for the model

		Parameters
		----------

		images: np.array[int]
			numpy array of the images, the images must be on the same size \
			The dimension of the matrix must be (n_images,width, deepth, channels)

		Returns
		----------
		images: np.array[int]
			numpy array of the images with the corresponding dimensions and size \
			to input the model (n_groups, 5, width, deepth, channels)
		"""
		images = tf.image.resize(images, (256, 256)).numpy()
		quantity = len(images)
		group_samples = len(images)//5
		images_groups = images[:group_samples*5].reshape((group_samples, 5)+images.shape[1:])
		images_left = images[group_samples*5:]
		if len(images_left)>0:
			images_left = self.generate_five_images(images_left)
			images_groups = np.concatenate([images_groups,images_left])
		return images_groups
	
	def generate_five_images(self, images):
		"""
		Generate a group of five images on a group of less number selecting \
		randomly from the actual images on the group

		Parameters
		----------

		images: np.array[int]
			numpy array of the images, the images must be on the same size \
			The dimension of the matrix must be (n_images,width, deepth, channels)

		Returns
		----------
		images: np.array[int]
			numpy array of the images with the corresponding dimensions and size \
			to input the model (1 , 5, width, deepth, channels)
		"""
		idxs = [i for i in range(len(images))]
		selected = sorted(list(np.random.choice(idxs, 5 - len(images))) + idxs)
		return np.expand_dims(images[selected,:],axis=0)

	def read_json(self,file):
		with open(file,'r') as f:
			config = json.loads(f.read())
		return config




class VGG16Labeler(Preprocess):
	
	CONFIG_PATH = 'models/config.ini'
	
	def __init__(self, model_type = 'frontal', device = '/device:CPU:0', verbose = 0):
		"""
		This class allows to run models to identify iRAP elements on streets in order to make the iRAP clasification 
		based on images every 20 metters. There are models that work with the lateral images, and others with the frontal image

		Parameters
		----------

		model type: str
			The route to the config file of the model
			
		device: str (default '/device:CPU:0')
			The name of the device that will be running the models.
			'/device:CPU:0' if you want to run them on the cpu
			'/device:GPU:0' if you have 1 GPU available and one to use all its resources
			If you have more than one GPU, you can select the number of the device, also \
			you can use it's power combined or reduce the resources of the GPU you want to use.\
			To do so reffer to the tensorflow documentation on https://www.tensorflow.org/guide/gpu
	
		verbose: int (default 0)
			Select the level of string information you want to be printed on the screen while running the process
			0: All the information
			Any other: No printing
		"""
		self.model_type = model_type
		self.device = device
		self.verbose = verbose
		if self.verbose==0:
			print('Configuration Loaded')
		self._load_config(self.CONFIG_PATH)		
		self._load_multi_model()
		if self.verbose == 0:
			print(f'You have succesfully load {len(self.models)} models on the category "{self.model_type}"')
		
		
	def _load_config(self, config_path):
		"""
		Function to load data and parameters from the models
		They can be frontal models (when the image is in the front of the vehicule) \
		or lateral model (when the camera was pointed to the lateral of the vehicle)


		Parameters
		----------

		config_path: str
			The route to the config file of the model

		"""
		if self.model_type not in ['lateral','frontal']:
			raise NameError(f'The model type "{self.model_type}" is not defined')
		options = {
			'frontal':'frontal_models',
			'lateral':'lateral_models',
		}
		config = configparser.ConfigParser()
		config.read(config_path)
		models =  json.loads(config.get('models',options[self.model_type]))
		clases = {}
		thresholds = {}
		model_class = {}
		for model in models:
			clases[model] = json.loads(config.get('clases',model))
			clases[model] = {int(k):v for k,v in clases[model].items()}
			thresholds[model] = json.loads(config.get('thresholds',model))
			if len(thresholds[model].keys())>0:
				thresholds[model] = {int(k):float(v) for k,v in thresholds[model].items()}
			else:
				thresholds[model] = None
			model_class[model] = json.loads(config.get('models_class',model))
		models_route = config.get('paths','models_route')
		self.models = models
		self.clases = clases
		self.thresholds = thresholds
		self.models_route = models_route
		self.model_class = model_class
		
		
		
	def _load_single_model(self, model_route, model_name):
		"""
		Load a single model to perform predictions


		Parameters
		----------

		model_route: str
			Route of the model

		model_type: str
			Name of the model


		Returns
		----------
		tf.keras.models.Model
			Model Type object
		"""
		with tf.device(self.device):
			model = tf.keras.models.load_model(model_route)
			input_m = tf.keras.layers.Input((5,256,256,3))
			output = model(input_m)
			_model = tf.keras.models.Model(input_m, output, name=model_name)
		return _model

	def _load_multi_model(self):
		"""
		Load all the models on the configuration file

		Parameters
		----------

		models: list[str]
			List of all the models available for loading

		models_route: str
			path to the model 


		Returns
		----------
		tf.keras.models.Model
			Model Type object
		"""
		with tf.device(self.device):
			input_model = tf.keras.layers.Input((5,256,256,3))
			models_artifacts = []
			for m in self.models:
				models_artifacts.append(self._load_single_model(self.models_route+m+'.h5', m))
				if self.verbose==0:
					print(f'Loaded model "{m}"')
			outputs = []
			for m in models_artifacts:
				outputs.append(m(input_model))
			self.model = tf.keras.models.Model(input_model, outputs)
	
	def get_labels(self, images):
		"""
		This function scores the images using the ML model

		images: np.array[int]
			numpy array of the images, the images must be on the same size \
			The dimension of the matrix must be (n_images,width, deepth, channels)

		Returns
		----------
		dict[] 
		raw_predictions: probabilities to belong an especific class for each model
		numeric_class: numeric class clasification taking under consideration the thresshold
		clasification: class name result ofr every group of images on every model
		
		"""
		if type(images)!= type(np.array([])):
			raise TypeError(f'This function only allows numpy arrays')
		if len(images.shape)!=4:
			raise TypeError(f'The shape of the images is {len(images.shape)} and this function only allows dimension 4')
		images = self.get_image_groups(images)
		predictions = self.get_raw_labels(images)
		results, class_results = self.get_discrete_value(predictions)
		return {'raw_predictions':predictions, 'numeric_class':results, 'clasification':class_results}
	
	def get_raw_labels(self, images):
		"""
		Scores the data points from the images using the diferent models inside \
		the Model object
		
		Parameters
		----------

		images: np.array[int]
			numpy array of the images with the corresponding dimensions and size \
		to input the model (n_groups , 5, width, deepth, channels)

		Returns
		----------
		dict[] np.array shape(n_groups,n_clases)
			A dictionary with the model name as key and an array of doubles as \
			values with the probability to belong for all of the clases
		"""
		with tf.device(self.device):
			pred = self.model.predict(images)
		results = {}
		for i in range(len(self.models)):
			results[self.models[i]] = pred[i]
		return results
	
	def get_discrete_value(self, predictions):
		"""
		Returns the specific label for all the clases depending on the scores 
		obtained
		
		Parameters
		----------

		predictions: dict[] np.array shape(n_groups,n_clases)
			A dictionary with the model name as key and an array of doubles as \
			values with the probability to belong for all of the clases

		Returns
		----------
		dict[] list
			For all the models, a list of the labels for every image group on the 
			input
		"""
		class_results = {}
		results = {}
		for k,v in predictions.items():
			if self.model_class[k]=='softmax':
				formal_clases = np.argmax(v,axis=1)
				if self.thresholds[k]:
					thresholds = np.vectorize(self.thresholds[k].get)(formal_clases)
					formal_clases = list(np.where(np.max(v, axis=1)>thresholds, formal_clases, -1))
				class_results[k] = list(np.vectorize(self.clases[k].get)(formal_clases))
			elif self.model_class[k]=='binary':
				th = 0.5
				if self.thresholds[k]:
					th = self.thresholds[k][1]
				if self.clases[k].get(0,None):
					formal_clases = np.where(v.reshape(-1)>th,1,0)
				else:
					formal_clases = np.where(v.reshape(-1)>th,1,-1)
			results[k] = formal_clases
			class_results[k] = list(np.vectorize(self.clases[k].get)(formal_clases))
				
		return results, class_results  
