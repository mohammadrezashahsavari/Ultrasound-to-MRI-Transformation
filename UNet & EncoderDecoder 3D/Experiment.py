import os
import sys
from Experiments.Transformation import *

# ============================================= Refrences ============================================= 
# UNet implemented based on 3D version of: https://arxiv.org/abs/1505.04597
# Source code from: https://www.kaggle.com/code/kmader/unet-conv3d-baseline


# EncoderDecoder implemented based on 3D version of: https://doi.org/10.1038/s41598-021-93747-y
# source code from (rewrote in tensorflow): https://github.com/bnl/CNN-Encoder-Decoder

# =====================================================================================================



# =====================================================================================================
# ========================================== USER PARAMETERS ========================================== 
''' -------- Tunable --------  '''
''' Parameters in this section should be tuned by the user to obtain the best results'''

# Which model do you want to use? choose among: 'EncoderDecoder', 'UNet'
network_structure = 'UNet'    

# Learning rate decay parameters. learning starts with learning rate value of 'initial_learning_rate' and after processing 'decay_steps' batches it will be decreased by the factor of 'decay_rate'.
initial_learning_rate= 0.00001
decay_steps=1000
decay_rate=0.96

# Number of epochs the model should be trained and the batch size you want the model to process in each step of the training.
max_epochs = 1000
batch_size = 2

# Used for train val test split. Setting this parameter ensures that each time you run the program, the same data will be placed in train val and test sets. Changing this parameter results in shuffling the data and creating a new order.
random_seed = 0

# train validation test split. train_ratio will automaticlly be set to 1 - (val_ratio + test_ratio)
val_ratio = 0.2
test_ratio = 0.7

# This parameter determines whether you want to train your model or generate outputs using a pre-trained model.
# It has to be set either on 'train' or 'generate_outputs'
mode = 'generate_outputs'

# =====================================================================================================
# =====================================================================================================

base_project_dir = os.path.abspath(os.path.dirname(__file__))  


args = {
    'initial_lr' : initial_learning_rate,
    'decay_steps' : decay_steps,
    'decay_rate' : decay_rate,
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

