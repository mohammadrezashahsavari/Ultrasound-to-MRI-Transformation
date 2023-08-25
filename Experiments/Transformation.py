import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from Tools import metrics
from Tools import callbacks
from batchup import data_source
import nibabel as nib
from Models.Pix2Pix2D import Pix2Pix
import utils
from pathlib import Path
import matplotlib.pyplot as plt


class Experiment_transformation():
    def __init__(self, preprocessed_dataset_path, network_structure='UNet', base_project_dir='.', args={}, mode='train'):
        self.preprocessed_dataset_path = preprocessed_dataset_path
        self.network_structure = network_structure
        self.base_project_dir = base_project_dir
        self.args = args
        self.task = ''
        self.mode = mode

        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)

    def prepare(self):

        self.train_dataset, self.val_dataset, self.test_dataset = utils.create_dataset(self.preprocessed_dataset_path, self.args['val_ratio'], self.args['test_ratio'], True, self.args['random_seed'])

        example_us, _ = next(iter(self.test_dataset))
        self.args['input_shape'] = list(example_us.shape[:2] ) + [1]
        print('Input Shape:', self.args['input_shape'])
        print('\nTrain size:', len(self.train_dataset))
        print('Val size:', len(self.val_dataset))
        print('Test size:', len(self.test_dataset))

        self.task += self.network_structure 

        output_dir = os.path.join(self.base_project_dir, 'Output', self.task)
        output_models_dir = os.path.join(output_dir, 'TrainedModels')
        self.args['best_models_dir'] = os.path.join(output_models_dir, 'BestModels')
        self.args['last_models_dir']= os.path.join(output_models_dir, 'LastModels')
        self.args['log_dir'] = os.path.join(output_dir, 'Log')
        self.args['sample_generated_images_dir'] = os.path.join(output_dir, 'SampleGeneratedImages')
        Path(self.args['best_models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.args['last_models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.args['sample_generated_images_dir']).mkdir(parents=True, exist_ok=True)

    
        if self.network_structure == 'Pix2Pix':
            self.model = Pix2Pix(args=self.args)

        print('Generator:')
        self.model.generator.summary()
        print('\n\nDiscriminator:')
        self.model.discriminator.summary()

        print("Task: " + self.task)

        
    def train(self):

        print("[INFO] Pix2Pix training initiated ...")

        self.model.fit(
            train_data = self.train_dataset,
            test_data = self.val_dataset
        )


