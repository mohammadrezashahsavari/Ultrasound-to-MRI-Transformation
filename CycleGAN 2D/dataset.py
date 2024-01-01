import numpy as np
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt


def multi_folder_glob(dirs, glob_pattern):
    """
    runs glob in a list of folder and merges their results
    """
    files = []
    for dir in dirs:
        files.extend(glob.glob(os.path.join(dir, glob_pattern)))
    return files


def load_sample(file_path):
    print(file_path)
    data = np.load(file_path.numpy())  # convert Tensor to numpy
    print(file_path.numpy())
    return data['us'], data['mri']


def create_dataset(path, train_ratio=0.75, val_ratio=0.1, test_ratio=0.15, shuffle=True, seed=None):
    # if sum([train_ratio, val_ratio, test_ratio]) == 1.0:
    #     raise ValueError()

    # get all the subject ids
    subjects = glob.glob(os.path.join(path, "*"))
    if shuffle:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(subjects)

    n_subjects = len(subjects)
    n_train = int(train_ratio * n_subjects)
    n_val = int(val_ratio * n_subjects)
    # n_test = n_subjects - n_train - n_val

    train_id_dirs = subjects[:n_train]
    val_id_dirs = subjects[n_train:n_train + n_val]
    test_id_dirs = subjects[n_train + n_val:]

    train_npz_files = multi_folder_glob(train_id_dirs, "*.npz")
    val_npz_files = multi_folder_glob(val_id_dirs, "*.npz")
    test_npz_files = multi_folder_glob(test_id_dirs, "*.npz")

    # suffeling the data
    np.random.seed(seed)
    np.random.shuffle(train_npz_files)
    np.random.seed(seed)
    np.random.shuffle(val_npz_files)
    np.random.seed(seed)
    np.random.shuffle(test_npz_files)
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train_npz_files)
    train_dataset = train_dataset.map(
        lambda file_path: tf.py_function(load_sample, [file_path], (tf.float32, tf.float32))
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(val_npz_files)
    val_dataset = val_dataset.map(
        lambda file_path: tf.py_function(load_sample, [file_path], (tf.float32, tf.float32))
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(test_npz_files)
    test_dataset = test_dataset.map(
        lambda file_path: tf.py_function(load_sample, [file_path], (tf.float32, tf.float32))
    )

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":

    base_project_dir = os.path.abspath(os.path.dirname(__file__))  
    dict_dataset_path = os.path.join(base_project_dir, 'Data', 'Preprocessed')

    train, val, test = create_dataset(dict_dataset_path, seed=20)

    for us, mri in train:
        plt.subplot(1, 2, 1)
        plt.imshow(us)
        plt.subplot(1, 2, 2)
        plt.imshow(mri)
        plt.show()
