import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from viasegura.configs.utils import DEFAULT_CONFIG_PATH
from viasegura.downloader import Downloader
from viasegura.utils.lanenet import loss_instance

viasegura_path = Path(__file__).parent
DEFAULT_DEVICE = "/device:CPU:0"


class Preprocess:
    def __init__(self):
        """
        This is the base preprocessing function but at the instance moment does not have any
        initialization process
        """
        pass

    def get_image_groups(self, images, batch_size=32):
        """
        Generate the grouped images for the model

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
        imgs = []
        for offset in range(0, images.shape[0], batch_size):
            imgs.append(tf.image.resize(images[offset : offset + batch_size], (256, 256)).numpy())
        images = np.concatenate(imgs)
        group_samples = len(images) // 5
        images_groups = images[: group_samples * 5].reshape((group_samples, 5) + images.shape[1:])
        images_left = images[group_samples * 5 :]
        if len(images_left) > 0:
            images_left = self.generate_five_images(images_left)
            images_groups = np.concatenate([images_groups, images_left])
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
        return np.expand_dims(images[selected, :], axis=0)

    def read_json(self, file):
        with open(file, "r") as f:
            config = json.loads(f.read())
        return config


class ModelLabeler(Preprocess):

    def __init__(
        self,
        model_type="frontal",
        model_filter=None,
        device=DEFAULT_DEVICE,
        config_path="config.json",
        system_path=viasegura_path,
        verbose=0,
    ):
        """
        This class allows to run models to identify iRAP elements on streets in order to make the iRAP classification
        based on images every 20 meters. There are models that work with the lateral images, and others with the frontal
        image.

        Parameters
        ----------
        model_type: str
            The camera direction which the photo was taken. It can be "frontal" or "lateral"

        model_filter: list[] str (default None)
            List with the models that will be used on the instance of the labeler. If this is None then it will use
            all of the same ModelType.
            Example: ['delineation','carriageway','street_lighting']

        config_path: str (default recomended 'config.json')
            The route to the file which contains the configuration of the package. To change it you must have a
            diferent config file. If you don't have an specific use case for change this option use Default

        system path: str (default Library instalation path)
            The route to the config file and to the artifacts that will be used by the labeler. This is the route
            where the models will be load from. In this route must be the config file as well. If you don't have and
            specific use case for this use Default.

        config_path type: str
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

        Properties
        ----------
        models: list
            The models loaded on the instance of the class

        classes: dict
            The classes for each model loaded on the instance

        thresholds: dict
            Specific thresholds for each model loaded on the instance

        model: tensorflow.python.keras.engine.functional.Functinal
            Object with the model in charge to calculate the scores for the images for each submodel

        Methods
        -------
        get_labels:
            Receive the images, groups them, scores them and determine the final class

        get_raw_labels:
            Receive the group of images and scores them

        get_discrete_value:
            Receive the scores of the groups and determines the final class
        """
        self.system_path = Path(system_path)
        self.config_path = DEFAULT_CONFIG_PATH / Path(config_path)

        self.downloader = Downloader(self.system_path / "models" / "models_artifacts")
        self.downloader.check_artifacts()

        self.model_filter = model_filter
        self.model_type = model_type
        self.device = device
        self.verbose = verbose
        self._load_config(self.config_path)
        if self.verbose == 0:
            print("Configuration Loaded")
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
        if self.model_type not in ["lateral", "frontal"]:
            raise NameError(f'The model type "{self.model_type}" is not defined')
        config = self.read_json(config_path)
        self.models_route = Path(config["paths"]["models_route"])
        self.models = config["models"][self.model_type]

        if self.model_filter:
            model_filter = set(self.model_filter)
            models_fake = model_filter - set(self.models)
            if len(models_fake) > 0:
                raise NameError(
                    f'The model(s) "{list(models_fake)}" are not defined, the valid option for models filter are {list(self.models)}'
                )
            self.models = list(model_filter.intersection(self.models))

        self.classes = {}
        self.thresholds = {}
        self.model_class = {}
        for model in self.models:
            if not self.downloader.check_files(self.system_path / self.models_route / (model + ".json")):
                raise ImportError(
                    f"The artifacts for the model {model} are not present use viasegura.download_models function to download them propertly"
                )
            model_config = self.read_json(self.system_path / self.models_route / (model + ".json"))
            self.classes[model] = model_config["classes"]
            self.classes[model] = {int(k): v for k, v in self.classes[model].items()}
            self.thresholds[model] = model_config["thresholds"]
            if len(self.thresholds[model].keys()) > 0:
                self.thresholds[model] = {int(k): float(v) for k, v in self.thresholds[model].items()}
            else:
                self.thresholds[model] = None
            self.model_class[model] = model_config["class"]

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
            if not self.downloader.check_files(self.system_path / model_route):
                raise ImportError(
                    f"The artifacts for the model {model_name} are not present use viasegura.download_models function "
                    + "to download them properly"
                )
            model = tf.keras.models.load_model(self.system_path / model_route)
            input_m = tf.keras.layers.Input((5, 256, 256, 3))
            output = model(input_m)
            _model = tf.keras.models.Model(input_m, output, name=model_name)
        return _model

    def _load_multi_model(self):
        """
        Load all the models on the configuration file

        Returns
        ----------
        tf.keras.models.Model
                Model Type object
        """
        with tf.device(self.device):
            input_model = tf.keras.layers.Input((5, 256, 256, 3))
            models_artifacts = []
            for m in self.models:
                models_artifacts.append(self._load_single_model(self.models_route / Path(m + ".h5"), m))
                if self.verbose == 0:
                    print(f'Loaded model "{m}"')
            outputs = []
            for m in models_artifacts:
                outputs.append(m(input_model))
            self.model = tf.keras.models.Model(input_model, outputs)

    def get_labels(self, images, batch_size=2):
        """
        This function scores the images using the ML model

        images: np.array[int]
            numpy array of the images, the images must be on the same size \
            The dimension of the matrix must be (n_images,width, deepth, channels)

        batch_size: int (default 2)
            Number of groups of images that can be scored at the same time. This number can cause a problem on the
            execution depending on the resources available for scoring.
            When you have low number of models loaded you can increase the batch size, but if the number is high we
            recomend (depending on the GPU and resources assigned to the model) to decrease this number.
            The default value has been tested on a NVIDIA RTX2070 with no issues, so we recommend to use this number
            if you have a GPU with similar memory available

        Returns
        ----------
        dict[]
        raw_predictions: probabilities to belong an especific class for each model
        numeric_class: numeric class clasification taking under consideration the thresshold
        clasification: class name result ofr every group of images on every model

        """
        if not isinstance(images, np.ndarray):
            raise TypeError("This function only allows numpy arrays")
        if len(images.shape) != 4:
            raise TypeError(f"The shape of the images is {len(images.shape)} and this function only allows dimension 4")
        images = self.get_image_groups(images) / 255.0
        predictions = self.get_raw_labels(images, batch_size=batch_size)
        results, class_results = self.get_discrete_value(predictions)
        return {"raw_predictions": predictions, "numeric_class": results, "clasification": class_results}

    def get_raw_labels(self, images, batch_size=2):
        """
        Scores the data points from the images using the diferent models inside \
        the Model object

        Parameters
        ----------

        images: np.array[int]
            numpy array of the images with the corresponding dimensions and size \
        to input the model (n_groups , 5, width, deepth, channels)

        batch_size: int (default 2)
            Number of groups of images that can be scored at the same time. This number can cause a problem on the
            execution depending on the resources available for scoring.
            When you have low number of models loaded you can increase the batch size, but if the number is high we
            recomend (depending on the GPU and resources assigned to the model) to decrease this number.
            The default value has been tested on a NVIDIA RTX2070 with no issues, so we recommend to use this number
            if you have a GPU with similar memory available


        Returns
        ----------
        dict[] np.array shape(n_groups,n_clases)
            A dictionary with the model name as key and an array of doubles as \
            values with the probability to belong for all of the clases
        """
        pred = [[] for _ in self.models]
        for offset in range(0, images.shape[0], batch_size):
            with tf.device(self.device):
                batch_pred = self.model.predict(images[offset : offset + batch_size], verbose=self.verbose)
                if len(self.models) == 1:
                    batch_pred = [batch_pred]

                for i, m in enumerate(self.models):
                    pred[i].append(batch_pred[i])

        for i, m in enumerate(self.models):
            pred[i] = np.concatenate(pred[i])

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
        for k, v in predictions.items():
            if self.model_class[k] == "softmax":
                formal_classes = np.argmax(v, axis=1)
                if self.thresholds[k]:
                    thresholds = np.vectorize(self.thresholds[k].get)(formal_classes)
                    formal_classes = list(np.where(np.max(v, axis=1) > thresholds, formal_classes, -1))
                class_results[k] = list(np.vectorize(self.classes[k].get)(formal_classes))
            elif self.model_class[k] == "binary":
                th = 0.5
                if self.thresholds[k]:
                    th = self.thresholds[k][1]
                if self.classes[k].get(0, None):
                    formal_classes = np.where(v.reshape(-1) > th, 1, 0)
                else:
                    formal_classes = np.where(v.reshape(-1) > th, 1, -1)
            results[k] = formal_classes
            class_results[k] = list(np.vectorize(self.classes[k].get)(formal_classes))

        return results, class_results


class LanesLabeler(Preprocess):
    CONFIG_PATH = "config.json"

    def __init__(self, lanenet_device=DEFAULT_DEVICE, models_device=DEFAULT_DEVICE, system_path=viasegura_path, verbose=0):
        """
        This class allows to run models to identify iRAP elements based on the implementation of Lanenet model
        (which allows to identify the delineation marks that divide channels on the street)
        Lanenet model create a mask with the delineation of the streets and mask the original image to apply further
        models over it.
        Based on the masked iamge, the object scores the groups images with models created for masked images based on
        the iRAP specifications.
        This model use 2 levels so it needs to specify 2 devices to run the models

        Parameters
        ----------

        lanenet_device: str (Default '/device:CPU:0')
            The name of the device that will be running the lanenet model.
            '/device:CPU:0' if you want to run them on the cpu
            '/device:GPU:0' if you have 1 GPU available and one to use all its resources
            If you have more than one GPU, you can select the number of the device, also \
            you can use it's power combined or reduce the resources of the GPU you want to use.
            It is recomended to use the logical gpus separation to run the models as explained on Tensorflow
            documentation
            To do so reffer to the tensorflow documentation on https://www.tensorflow.org/guide/gpu

        models_device: str (Default '/device:CPU:0')
            The name of the device that will be running the model masked image models.
            '/device:CPU:0' if you want to run them on the cpu
            '/device:GPU:0' if you have 1 GPU available and one to use all its resources
            If you have more than one GPU, you can select the number of the device, also \
            you can use it's power combined or reduce the resources of the GPU you want to use.
            It is recomended to use the logical gpus separation to run the models as explained on Tensorflow
            documentation
            To do so reffer to the tensorflow documentation on https://www.tensorflow.org/guide/gpu

        config_path: Path (default recomended 'config.json')
            The route to the file which contains the configuration of the package. To change it you must have a diferent
            config file. If you don't have an specific use case for change this option use Default


        verbose: int (default 1)
            Select the level of string information you want to be printed on the screen while running the process
            1: All the information
            0: No printing

        Properties
        ----------

        Labeler:
            An instance from ModelLabeler with the models created for masked lanenet images.
            Refer to ModelLabeler documentation to see inputs, properties and methods

        lanenet: list
            Model that creates the masked image


        Methods
        -------

        get_labels:
            Receive the images, create masks and scores the final models classes

        get_mask_images:
            Receive the iamges and transform them into masked images

        """
        self.lanenet_device = lanenet_device
        self.models_device = models_device
        self.system_path = Path(system_path)
        self.verbose = verbose

        self.downloader = Downloader(self.system_path / "models" / "models_artifacts")
        self.downloader.check_artifacts()

        self.load_config()
        self.load_lanenet_model()
        self.labeler = ModelLabeler(
            system_path=system_path, model_type="frontal", device=self.models_device, config_path=viasegura_path / "lanenet_config.json"
        )
        if self.verbose == 0:
            print("Lanenet model loaded successfully")

    def load_config(self):
        """
        Function to load data and parameters from the models
        This function only loads the configuration to the lanenet model. The configuration for the models \
        to the masked iamges are on the ModelLabeler instance inside the LanesLabeler with its own configuration file

        """
        config = self.read_json(DEFAULT_CONFIG_PATH / self.CONFIG_PATH)
        self.models_route = Path(config["paths"]["models_route_lanenet"])
        self.img_shape = tuple([int(n) for n in config["lanenet"]["input_shape"]])

    def load_lanenet_model(self):
        """
        Function to load the lanenet model on the instance using the corresponding device

        """

        with tf.device(self.lanenet_device):
            if not self.downloader.check_files(self.system_path / self.models_route / "lanenet.h5"):
                raise ImportError(
                    "The artifacts for the model lanenet are not present use viasegura.download_models function to download them propertly"
                )
            self.lanenet = tf.keras.models.load_model(
                self.system_path / self.models_route / "lanenet.h5", custom_objects={"loss_instance": loss_instance}
            )

    def get_labels(self, images, batch_size=4):
        """
        Scores the data points from the images using the diferent models inside the instance

        Parameters
        ----------

        images: np.array[int]
            numpy array of the images with the corresponding dimensions and size \
        to input the model (n_groups , 5, width, deepth, channels)

        batch_size: int (default 4)
            Number of images that can be scored at the same time. This number can cause a problem on the execution
            depending on the resources available for scoring.
            The default value has been tested on a NVIDIA RTX2070 with no issues, so we recommend to use this number
            if you have a GPU with similar memory available


        Returns
        ----------
        dict[] np.array shape(n_groups,n_clases)
            A dictionary with the model name as key and an array of doubles as \
            values with the probability to belong for all of the clases
        """

        mask_images = []
        for offset in range(0, images.shape[0], batch_size):
            mask_images.append(self.get_mask_images(images[offset : offset + batch_size]))
        mask_images = np.concatenate(mask_images)
        if self.verbose == 1:
            print("Mask images generated sucessfully")
        return self.labeler.get_labels(mask_images, batch_size=batch_size)

    def get_mask_images(self, images):
        """
        Use the lanenet model to get the mask image for every image

        Parameters
        ----------

        images: np.array[int]
            numpy array of the images with the corresponding dimensions and size \
            to input the model (n_groups , 5, width, deepth, channels)

        Returns
        ----------
        np.array shape(n_iamges,widht, height, channels)
            Images masked and transformed

        """
        images = tf.image.resize(images, self.img_shape[:2]).numpy() / 255.0
        with tf.device(self.lanenet_device):
            pred = self.lanenet.predict(images, verbose=self.verbose)
        binary_images = pred[0]
        binary_processed_images = np.array(list(map(lambda bin_img: self._postprocess(bin_img), binary_images)))
        transformed_images = (images.copy() * 255).astype("uint8")
        transformed_images[:, :, :, 2] = binary_processed_images
        return transformed_images

    def _postprocess(self, binary_output, min_area_threshold=100, threshold=0.15):
        """
        Postprocess for the binary output of the mask image

        Parameters
        ----------
        binary_output: np.array[int]
                numpy array of the image with the corresponding dimensions and size (width, deepth)

        """

        binary_image_transformed = np.zeros(binary_output.shape)
        binary_image_transformed[binary_output > threshold] = 255
        binary_image_transformed = binary_image_transformed.astype("uint8")
        morphological_ret = self._morphological_process(binary_image_transformed)
        _ = self._connect_components_analysis(morphological_ret)
        return morphological_ret

    def _morphological_process(self, image, kernel_size=5):
        """
        morphological process to fill the hole in the binary segmentation result
        :param image:
        :param kernel_size:
        :return:
        """
        if len(image.shape) == 3:
            raise ValueError("Binary segmentation result image should be a single channel image")

        if image.dtype is not np.uint8:
            image = np.array(image, np.uint8)

        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

        # close operation fille hole
        closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

        return closing

    def _connect_components_analysis(self, image):
        """
        connect components analysis to remove the small components
        :param image:
        :return:
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)
