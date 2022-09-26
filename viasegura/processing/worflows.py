import os
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from viasegura.processing.img_sources import Image_Source_Loader
from viasegura.processing.gps_sources import GPS_Data_Loader
import tensorflow as tf
from tqdm.autonotebook import tqdm

class Workflow_Processor:
    def __init__(self, images_input, **kwargs):
        image_source_type = kwargs.get('image_source_type', 'video')
        gps_source_type = kwargs.get('gps_source_type', 'loc')
        gps_in = kwargs.get('gps_input', images_input if gps_source_type == image_source_type else None)
        adjust_gps = kwargs.get('adjust_gps', True)
        gps_sections_distance = kwargs.get('gps_sections_distance', 100)

        self.img_obj = img_obj = Image_Source_Loader(image_source_type, images_input)
        self.gps_data = GPS_Data_Loader(gps_source_type,gps_in)
        if adjust_gps:
            if image_source_type =='video':
                self.gps_data.adjust_gps_data(self.img_obj.number_of_frames)
            else:
                self.gps_data.adjust_gps_data(self.img_obj.get_len())
        self.gps_data.generate_gps_metrics(gps_sections_distance)
        self.img_obj.update_obj_inf(self.gps_data.image_numbers)
        self.executed = False
    
    def execute(self, labeler, return_results = True):
        self.raw_predictions, self.numeric_class, self.classification = labeler.get_labels(self.img_obj, batch_size = 3)
        classification_df = pd.DataFrame(self.classification)
        classification_df['section_id'] = classification_df.index
        for col in self.raw_predictions.keys():
            classification_df[f'{col}_confidence'] = np.array(list(map(lambda x, y: x[y], self.raw_predictions['delineation'],self.numeric_class['delineation'])))
        classification_df = classification_df[sorted(classification_df.columns)]
        self.result_df = pd.merge(self.gps_data.gps_group_info, classification_df, on= 'section_id', how = 'left')
        self.executed = True
        if return_results:
            return self.result_df, self.raw_predictions, self.numeric_class, self.classification
    
    def get_results(self):
        if self.executed:
            return self.result_df, self.raw_predictions, self.numeric_class, self.classification
        else:
            raise ValueError(f'Workflow not yet executed, use execute method to store the results after executing models')