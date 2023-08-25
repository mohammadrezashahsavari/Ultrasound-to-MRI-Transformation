import os
import sys
from Experiments.Transformation import *

# =====================================================================================================
# ============================================= Refrences =============================================
# Implemetation of Pix2Pix network based on: https://arxiv.org/abs/1611.07004
# Pix2Pix source code from: https://www.tensorflow.org/tutorials/generative/pix2pix



# =====================================================================================================
# ========================================== USER PARAMETERS ========================================== 
''' -------- Tunable --------  '''
''' Parameters in this section should be tuned by the user to obtain the best results'''

# Which model do you want to use? choose among: 'CycleGAN'
network_structure = 'Pix2Pix'    

# Learning rate parameters. 'lr_D' is the discriminator's learning rate and 'lr_G' is the generator's learning rate
lr_D = 0.00001
lr_G = 0.00001

# Number of epochs the model should be trained and the batch size you want the model to process in each step of the training.
max_epochs = 1000
batch_size = 2

# After how many number of iteration do you want to save outputs?. 'save_weights_freq': save model weights after this iteration number. 'save_image_freq': save sample output image after this number of iteration 
save_weights_freq = 2
save_image_freq = 2


# Used for train val test split. Setting this parameter ensures that each time you run the program, the same data will be placed in train val and test sets. Changing this parameter results in shuffling the data and creating a new order.
random_seed = 0

# train validation test split. train_ratio will automaticlly be set to 1 - (val_ratio + test_ratio)
val_ratio = 0.1
test_ratio = 0.15

# number of subjects you want to load. in the case of facing memory limit errors you can reduce the number of subject data that will be loaded using this parameter. if you wat to load all the data set this parameter to -1.
use_n_subjects = -1

# This parameter determines whether you want to train your model or generate outputs using a pre-trained model.
# It has to be set either on 'train' or 'generate_outputs'
mode = 'train'

# =====================================================================================================
# =====================================================================================================

base_project_dir = os.path.abspath(os.path.dirname(__file__))  

preprocessed_dataset_path = os.path.join(base_project_dir, 'Data', 'Preprocessed')

args = {
    'lr_D' : lr_D,
    'lr_G': lr_G,
    'max_epochs' : max_epochs,
    'batch_size' : batch_size,
    'save_weights_freq' : save_weights_freq,
    'save_image_freq' : save_image_freq,
    'random_seed' : random_seed,
    'val_ratio' : val_ratio,
    'test_ratio' : test_ratio,
    'use_n_subjects' : use_n_subjects,
}


exp = Experiment_transformation(preprocessed_dataset_path, network_structure, base_project_dir, args, mode)
exp.prepare()
if mode == 'train':
    exp.train()
elif mode == 'generate_outputs':
    exp.generate_output()

