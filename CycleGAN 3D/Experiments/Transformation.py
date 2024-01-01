import os
import numpy as np
from Models.CycleGAN.cyclegan3d import CycleGAN
from Models.CycleGAN.losses import generator_loss
from tqdm import tqdm
import pandas as pd
import tensorflow as tf
from Tools import metrics
from Tools import callbacks
from batchup import data_source
import nibabel as nib
import utils
from pathlib import Path


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
        self.train_dataset, self.val_dataset, self.test_dataset, self.file_names = utils.create_dataset(self.preprocessed_dataset_path, self.args['val_ratio'], self.args['test_ratio'], True, self.args['random_seed'])

        # this is useful for 'generate_output' mode where we wnat to save data
        image_path = os.path.join(self.base_project_dir, 'Data', 'US_Images', '0001_US_Image_a.nii.gz')
        nii_img  = nib.load(image_path)
        self.images_metadata = {'affine':nii_img.affine, 'header':nii_img.header}

        example_us, _ = next(iter(self.test_dataset))
        self.args['input_shape'] =  example_us.shape #list(example_us.shape) + [1]
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
        self.args['sample_pred_image_path'] = os.path.join(self.base_project_dir, 'Output', self.task, 'SamplePredictedImages')

        if self.network_structure == 'CycleGAN':
            self.model = CycleGAN(args={'g_residual_blocks':self.args['g_residual_blocks'], 'lr_G':self.args['lr_G'], 'lr_D':self.args['lr_D']})

        if os.path.exists(os.path.join(self.args['last_models_dir'], 'G_A2B.h5')):
            print('Loading pretrained weights from:'  + self.args['last_models_dir'])
            self.model.G_A2B.load_weights(os.path.join(self.args['last_models_dir'], 'G_A2B.h5'))
            self.model.G_B2A.load_weights(os.path.join(self.args['last_models_dir'], 'G_B2A.h5'))
            self.model.D_A.load_weights(os.path.join(self.args['last_models_dir'], 'D_A.h5'))
            self.model.D_B.load_weights(os.path.join(self.args['last_models_dir'], 'D_B.h5'))
        else:
            print('Pretrained weights were not found in the best models directory. Starting to train with randomly initialized weights.')

        print('Generator A2B:')
        self.model.G_A2B.summary()
        print('\n\nGenerator B2A:')
        self.model.G_B2A.summary()
        print('\n\nDiscriminatore A:')
        self.model.D_A.summary()
        print('\n\nDiscriminatore B:')
        self.model.D_B.summary()

        Path(self.args['best_models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.args['last_models_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.args['sample_generated_images_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.args['sample_pred_image_path']).mkdir(parents=True, exist_ok=True)

        print("Task: " + self.task)

        
    def train(self):

        print("[INFO] CycleGAN training initiated ...")
        
        D_A_losses, D_B_losses, G_A2B_losses, G_B2A_losses, cycle_A_losses, cycle_B_losses = ([] for i in range(6))

        best_val_G_A2B_loss = float('inf')

        epoch = 0
        while epoch < self.args['max_epochs']:
            iteration = 1
            for (imgA, imgB) in self.train_dataset.batch(self.args['batch_size']):

                fake_A, fake_B, cycle_A, cycle_B, G_A2B_loss, G_B2A_loss, cycle_A_loss, cycle_B_loss, D_A_loss, D_B_loss = self.model.train_step(imgA, imgB)

                D_A_losses.append(D_A_loss.numpy())
                D_B_losses.append(D_B_loss.numpy())
                G_A2B_losses.append(G_A2B_loss.numpy())
                G_B2A_losses.append(G_B2A_loss.numpy())
                cycle_A_losses.append(cycle_A_loss.numpy())
                cycle_B_losses.append(cycle_B_loss.numpy())

                print('Iteration: ' + str(iteration))

                iteration += 1

            # save the last model
            self.model.G_A2B.save_weights(os.path.join(self.args['last_models_dir'], 'G_A2B.h5'))
            self.model.G_B2A.save_weights(os.path.join(self.args['last_models_dir'], 'G_B2A.h5'))
            self.model.D_A.save_weights(os.path.join(self.args['last_models_dir'], 'D_A.h5'))
            self.model.D_B.save_weights(os.path.join(self.args['last_models_dir'], 'D_B.h5'))

            pred_MRI_val = self.model.G_A2B(next(iter(self.val_dataset.batch(16))))
            val_G_A2B_loss = generator_loss(pred_MRI_val)
            if val_G_A2B_loss < best_val_G_A2B_loss:
                best_val_G_A2B_loss = val_G_A2B_loss
                self.model.G_A2B.save_weights(os.path.join(self.args['best_models_dir'], 'G_A2B.h5'))
                self.model.G_B2A.save_weights(os.path.join(self.args['best_models_dir'], 'G_B2A.h5'))
                self.model.D_A.save_weights(os.path.join(self.args['best_models_dir'], 'D_A.h5'))
                self.model.D_B.save_weights(os.path.join(self.args['best_models_dir'], 'D_B.h5'))
                print(f'Best generator vall loss achieved: val_G_A2B_loss: {val_G_A2B_loss}. Saving model weights.')

            if epoch % self.args['save_image_freq'] == 0:
                save_to = os.path.join(self.args['sample_pred_image_path'], 'PredMRI_epoch'+str(epoch)+'.nii.gz')
                image = np.squeeze(fake_B[0])
                pred_nii_img = nib.Nifti1Image(image, self.images_metadata['affine'], self.images_metadata['header'])
                nib.save(pred_nii_img, save_to)
            
            print('epoch: ' + str(epoch) + ' --- G_A2B_loss: ' + str(float(G_A2B_loss))[:6] + ' --- G_B2A_loss: ' + str(float(G_B2A_loss))[:6] + ' --- D_A_loss: ' + str(float(D_A_loss))[:6] + ' --- D_B_loss: ' + str(float(D_B_loss))[:6] + ' --- val_G_A2B_loss: ' + str(float(val_G_A2B_loss))[:6])
            epoch += 1


    def generate_output(self):
        output_path = os.path.join(self.base_project_dir, 'Output', self.task, 'Results')
        Path(output_path).mkdir(parents=True, exist_ok=True)
        for set_name in ['train', 'val', 'test']:
            for result_dir_name in ['GeneratedImages', 'DifferentialImages', 'ResultedMetrics']:
                result_dir_path = os.path.join(output_path, set_name, result_dir_name)
                Path(result_dir_path).mkdir(parents=True, exist_ok=True)

        print('Loading the best model weights from: ' + self.args['last_models_dir'])
        self.model.G_A2B.load_weights(os.path.join(self.args['last_models_dir'], 'G_A2B.h5'))

        print('Generating outputs ...')
        for set_name in ['train', 'val', 'test']:
            generated_images_path = os.path.join(output_path, set_name, 'GeneratedImages')
            differential_images_path = os.path.join(output_path, set_name, 'DifferentialImages')
            resulted_metrics_path = os.path.join(output_path, set_name, 'ResultedMetrics')
            
            df_columns_names = ['MAE', 'MAPE', 'RMSE', 'SSI']#, 'PSNR']
            df_metrics = pd.DataFrame(columns=df_columns_names)

            if set_name == 'train':
                dataset = self.train_dataset
            elif set_name == 'val':
                dataset = self.val_dataset
            elif set_name == 'test':
                dataset = self.test_dataset

            for i, (US, gt_MRI) in tqdm(enumerate(dataset)):
                file_name = self.file_names[set_name][i].split(os.sep)[-1].replace('.npz', '')
                US = US.numpy()
                gt_MRI = gt_MRI.numpy()

                pred_MRI = self.model.G_A2B(np.expand_dims(US, axis=0), training=True).numpy()
                pred_MRI = np.squeeze(pred_MRI)
                gt_MRI = np.squeeze(gt_MRI)

                pred_MRI_path = os.path.join(generated_images_path, file_name)
                pred_nii_img = nib.Nifti1Image(pred_MRI, self.images_metadata['affine'], self.images_metadata['header'])
                nib.save(pred_nii_img, pred_MRI_path)

                diff_MRI = gt_MRI - pred_MRI
                diff_MRI_path = os.path.join(differential_images_path, file_name)
                diff_nii_img = nib.Nifti1Image(diff_MRI, self.images_metadata['affine'], self.images_metadata['header'])
                nib.save(diff_nii_img, diff_MRI_path)

                mae = metrics.Mean_absolute_error(gt_MRI, pred_MRI)
                mape = metrics.Mean_absolute_percentage_error(gt_MRI, pred_MRI)
                rmse = metrics.Root_mean_squared_error(gt_MRI, pred_MRI)
                ssi = metrics.Structural_similarity(gt_MRI, pred_MRI)
                # psnr = metrics.Peak_signal_noise_ratio(gt_MRI, pred_MRI)
                # psnr = 0

                df_metrics.loc[file_name] = [mae, mape, rmse, ssi]#, psnr]
            
            df_metrics_path = os.path.join(resulted_metrics_path, 'CalculatedMetrics.csv')
            df_metrics.to_csv(df_metrics_path)



        




        