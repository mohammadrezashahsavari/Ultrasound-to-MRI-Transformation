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
network_structure = 'CycleGan'    

# Learning rate parameters. 'lr_D' is the discriminator's learning rate and 'lr_G' is the generator's learning rate
lr_gen_G = 2e-4
lr_gen_F = 2e-4
lr_disc_X = 2e-4
lr_disc_Y = 2e-4

# Number of epochs the model should be trained and the batch size you want the model to process in each step of the training.
max_epochs = 1000
batch_size = 1


# Used for train val test split. Setting this parameter ensures that each time you run the program, the same data will be placed in train val and test sets. Changing this parameter results in shuffling the data and creating a new order.
random_seed = 0

# train validation test split. train_ratio will automaticlly be set to 1 - (val_ratio + test_ratio)
val_ratio = 0.1
test_ratio = 0.15

# This parameter determines whether you want to train your model or generate outputs using a pre-trained model.
# It has to be set either on 'train', 'generate_outputs_2D' or 'generate_outputs_3D'.
mode = 'train'

# =====================================================================================================
# =====================================================================================================

base_project_dir = os.path.abspath(os.path.dirname(__file__))  

preprocessed_dataset_path = os.path.join(base_project_dir, 'Data', 'Preprocessed')

args = {
    'lr_gen_G' : lr_gen_G,
    'lr_gen_F': lr_gen_G,
    'lr_disc_X' : lr_disc_X,
    'lr_disc_Y' : lr_disc_Y,
    'max_epochs' : max_epochs,
    'batch_size' : batch_size,
    'random_seed' : random_seed,
    'val_ratio' : val_ratio,
    'test_ratio' : test_ratio,
}


exp = Experiment_transformation(preprocessed_dataset_path, network_structure, base_project_dir, args, mode)
exp.prepare()
if mode == 'train':
    exp.train()
elif mode == 'generate_outputs_2D':
    exp.generate_output_2D()
elif mode == 'generate_outputs_3D':
    exp.generate_output_3D()

