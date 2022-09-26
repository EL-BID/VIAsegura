import os
from pathlib import Path
import pandas as pd
import numpy as np
from viasegura.labelers import ModelLabeler, LanesLabeler
import tensorflow as tf
from tqdm.autonotebook import tqdm

class Images_Labeler:
    def __init__(self, processor_config_file = 'config.json', assign_devices = False, gpu_enabled = False, total_mem = 6144, 
                 frontal_device = '/device:CPU:0', lanes_device_sep = '/device:CPU:0', lanes_device_models = '/device:CPU:0', artifacts_path = None):

        self.assign_model_devices(assign_devices, gpu_enabled, total_mem, frontal_device, lanes_device_sep, lanes_device_models)
        self.frontal_model = ModelLabeler('frontal', device = self.frontal_device)
        self.lanes_labeler = LanesLabeler(lanenet_device = self.lanes_device_sep, models_device = self.lanes_device_models) 

    def assign_model_devices(self, assign_devices, gpu_enabled, total_mem, frontal_device, lanes_device_sep,lanes_device_models):
        if assign_devices == True:
            if gpu_enabled == True:
                self.assign_gpu_devices(total_mem)
            else:
                self.frontal_device = '/device:CPU:0'
                self.lanes_device_sep = '/device:CPU:0'
                self.lanes_device_models = '/device:CPU:0'
        else:
            self.frontal_device = frontal_device
            self.lanes_device_sep = lanes_device_sep
            self.lanes_device_models = lanes_device_models

    def assign_gpu_devices(self, total_mem):
        memory_unit = int(total_mem/12)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:   
            try:	
                tf.config.experimental.set_virtual_device_configuration(gpus[0], 
                                                                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8*memory_unit),
                                                                         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3*memory_unit),
                                                                         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1*memory_unit)])	
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')	
                print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")  
            except RuntimeError as e:
                print(e)
        self.frontal_device = logical_gpus[0].name
        self.lanes_device_sep = logical_gpus[1].name
        self.lanes_device_models = logical_gpus[2].name 
        
        
    def _get_dict(self, d, prediction):
        for k,v in prediction.items():
            d[k] = v
        return d

    def _get_total_dicts(self, raw_predictions,numeric_class,clasification,prediction):
        raw_predictions = self._get_dict(raw_predictions, prediction['raw_predictions'])
        numeric_class = self._get_dict(numeric_class, prediction['numeric_class'])
        clasification = self._get_dict(clasification, prediction['clasification'])
        return raw_predictions, numeric_class, clasification

    def _get_dict_mult(self, d1,d2):
        for k in d1.keys():
            d1[k] = np.concatenate([np.array(d1[k]), np.array(d2[k])])
        return d1

    def _get_total_dicts_mult(self, r_p, n_c, cl, raw_predictions,numeric_class,clasification):
        r_p = self._get_dict_mult(r_p, raw_predictions)
        n_c = self._get_dict_mult(n_c, numeric_class)
        cl = self._get_dict_mult(cl, clasification)
        return r_p, n_c, cl
        
    def get_labels(self, img_obj, batch_size = 5):
        len_imgs = img_obj.get_len_sel()
        r_p, n_c, cl = None, None, None
        for offset in tqdm(range(0,img_obj.get_len_sel(), batch_size*5)):
            raw_predictions = {}
            numeric_class = {}
            clasification = {}
            img_batch = img_obj.get_batch_sel(offset, batch_size*5)
            prediction_frontal = self.frontal_model.get_labels(img_batch, batch_size = batch_size)
            raw_predictions, numeric_class, clasification = self._get_total_dicts(raw_predictions, numeric_class, clasification, prediction_frontal)
            prediction_lanes = self.lanes_labeler.get_labels(img_batch, batch_size = batch_size)
            raw_predictions, numeric_class, clasification = self._get_total_dicts(raw_predictions, numeric_class, clasification, prediction_lanes)
            if r_p==None:
                r_p, n_c, cl = raw_predictions, numeric_class, clasification
            else:
                r_p, n_c, cl = self._get_total_dicts_mult(r_p, n_c, cl, raw_predictions,numeric_class,clasification)
        return r_p, n_c, cl