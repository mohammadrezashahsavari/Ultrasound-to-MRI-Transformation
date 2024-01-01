import os
import sys
from Experiments.Transformation import *

# ============================================= Refrences ============================================= 
# CycleGAN implemented based on 3D version of: https://arxiv.org/abs/1703.10593
# Source code from: https://github.com/rekalantar/PatchBased_3DCycleGAN_CT_Synthesis

# =====================================================================================================


# =====================================================================================================
# ========================================== USER PARAMETERS ========================================== 
''' -------- Tunable --------  '''
''' Parameters in this section should be tuned by the user to obtain the best results'''

# Which model do you want to use? choose among: 'CycleGAN'
network_structure = 'CycleGAN'    

# Learning rate parameters. 'lr_D' is the discriminator's learning rate and 'lr_G' is the generator's learning rate
lr_D = 0.00001
lr_G = 0.00001

# How many resudential blockes doese generatore have.
g_residual_blocks = 1

# Number of epochs the model should be trained and the batch size you want the model to process in each step of the training.
max_epochs = 1000
batch_size = 2

# After how many number of iteration do you want to save outputs?. 'save_weights_freq': save model weights after this iteration number. 'save_image_freq': save sample output image after this number of iteration 
save_weights_freq = 1
save_image_freq = 1

# Used for train val test split. Setting this parameter ensures that each time you run the program, the same data will be placed in train val and test sets. Changing this parameter results in shuffling the data and creating a new order.
random_seed = 0

# train validation test split. train_ratio will automaticlly be set to 1 - (val_ratio + test_ratio)
val_ratio = 0.5
test_ratio = 0.3

# This parameter determines whether you want to train your model or generate outputs using a pre-trained model.
# It has to be set either on 'train' or 'generate_outputs'
mode = 'train'

# =====================================================================================================
# =====================================================================================================


args = {
    'lr_D' : lr_D,
    'lr_G': lr_G,
    'save_weights_freq'  :save_weights_freq,
    'save_image_freq':save_image_freq,
    'g_residual_blocks' : g_residual_blocks,
    'max_epochs' : max_epochs,
    'batch_size' : batch_size,
    'random_seed' : random_seed,
    'val_ratio' : val_ratio,
    'test_ratio' : test_ratio
}

base_project_dir = os.path.abspath(os.path.dirname(__file__))  
preprocessed_dataset_path = os.path.join(base_project_dir, 'Data', 'Preprocessed')

exp = Experiment_transformation(preprocessed_dataset_path, network_structure, base_project_dir, args, mode)
exp.prepare()
if mode == 'train':
    exp.train()
elif mode == 'generate_outputs':
    exp.generate_output()

