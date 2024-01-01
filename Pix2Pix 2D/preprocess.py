from multiprocessing import Pool, cpu_count
import utils
import os
import nibabel as nib
import numpy as np
from tqdm import tqdm


def correct_orientation(ct_image, ct_arr):
    x, y, z = nib.aff2axcodes(ct_image.affine)
    if x != 'R':
        ct_arr = np.flip(ct_arr, axis=0)
    if y != 'P':
        ct_arr = np.flip(ct_arr, axis=1)
    if z != 'S':
        ct_arr = np.flip(ct_arr, axis=2)
    return ct_arr


def save_3d_as_2d_slices(mri, us, dist, basename):
    for z in range(mri.shape[2]):
        np.savez(os.path.join(dist, f"MRI_US_ID{basename}_SLICE{z:04d}"),
                 mri=mri[:, :, z],
                 us=us[:, :, z]
                 )


def process_image(args):
    us_image, mri_image, us_parent_path, mri_parent_path, dist_path, resize_to = args
    image_id = us_image.replace("US_Image_", "").replace(".nii.gz", "")

    us_image = os.path.join(us_parent_path, us_image)
    mri_image = os.path.join(mri_parent_path, mri_image)

    dist = os.path.join(dist_path, image_id)
    os.makedirs(dist, exist_ok=True)

    us_nii_img = nib.load(us_image)
    us_data = us_nii_img.get_fdata().astype(np.float32)
    us_data = us_data[1:]     # removing firt slice
    us_data = utils.min_max_normalize(us_data)
    us_data = utils.resize_volume(us_data, resize_to)
    us_data = np.expand_dims(us_data, axis=-1)

    mri_nii_img = nib.load(mri_image)
    mri_data = mri_nii_img.get_fdata().astype(np.float32)
    mri_data = mri_data[1:]    # removing firt slice
    mri_data = utils.min_max_normalize(mri_data)
    mri_data = utils.resize_volume(mri_data, resize_to)
    mri_data = np.expand_dims(mri_data, axis=-1)

    save_3d_as_2d_slices(mri_data, us_data, dist, image_id)


def dataset2npz(us_parent_path, mri_parent_path, dist_path, resize_to):
    us_images, mri_images = utils.get_usable_US_and_MRI_names(us_parent_path, mri_parent_path)
    args_list = []
    for us_image, mri_image in zip(us_images, mri_images):
        args_list.append((us_image, mri_image, us_parent_path,
                         mri_parent_path, dist_path, resize_to))

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap(process_image, args_list), total=len(args_list)))


if __name__ == "__main__":
    base_project_dir = os.path.abspath(os.path.dirname(__file__))  

    data_path = os.path.join(base_project_dir, 'Data')
    US_dataset_path = os.path.join(data_path, 'US_Images')
    MRI_dataset_path = os.path.join(data_path, 'MRI_Images')
    dict_dataset_path = os.path.join(data_path, 'Preprocessed')

    resize_to = [256, 256, 48]
    dataset2npz(US_dataset_path, MRI_dataset_path, dict_dataset_path, resize_to)

