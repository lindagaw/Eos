from __future__ import print_function

import pickle
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
import params

import os
import gzip
from torchvision import datasets, transforms

import sys
import os, sys, tarfile, errno
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib # ugly but works
else:
    import urllib

try:
    from imageio import imsave
except:
    from scipy.misc import imsave

print(sys.version_info)

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = './data'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
DATA_PATH = './data/stl10_binary/train_X.bin'

# path to the binary train file with labels
LABEL_PATH = './data/stl10_binary/train_y.bin'

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()

def save_image(image, name):
    imsave("%s.png" % name, image, format="png")

def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def save_images(images, labels):
    print("Saving images to disk")
    i = 0
    for image in images:
        label = labels[i]
        directory = './img/' + str(label) + '/'
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = directory + str(i)
        print(filename)
        save_image(image, filename)
        i = i+1


class STL_10(data.Dataset):

    def __init__(self, root, train=True, transform=None, download=False, dataset='undefined'):
        """Init USPS dataset."""
        self.root = 'data//stl-10//'
        try:
            os.makedirs(self.root)
        except:
            print('root already exists. skipping ...')

        self.training = dataset + ".pkl"
        self.testing = dataset + "_eval.pkl"
        self.train = train

        self.transform = transform
        self.dataset_size = None

        print('loading training data from ' + self.training)
        print('loading testing data from ' + self.testing)
        # download dataset.
        if download:
            # download data if needed
            download_and_extract()
            # test to check if the whole dataset is read correctly
            images = read_all_images(DATA_PATH)
            labels = read_labels(LABEL_PATH)

            xs_train, xs_test, ys_train, ys_test = train_test_split(images, labels, test_size=0.33, random_state=42)

            np.save(self.root + dataset + '//xs_train.npy', xs_train)
            np.save(self.root + dataset + '//xs_test.npy', xs_test)
            np.save(self.root + dataset + '//ys_train.npy', ys_train)
            np.save(self.root + dataset + '//ys_train.npy', ys_train)

            xs_train = torch.from_numpy(np.load(self.root + dataset + '//xs_train.npy'))
            xs_test = torch.from_numpy(np.load(self.root + dataset + '//xs_test.npy'))
            ys_train = torch.from_numpy(np.load(self.root + dataset + '//ys_train.npy'))
            ys_test = torch.from_numpy(np.load(self.root + dataset + '//ys_test.npy'))

            torch.save(TensorDataset(xs_train, ys_train), self.root + self.training)
            torch.save(TensorDataset(xs_test, ys_test), self.root + self.testing)

            data_set_train = torch.load(self.root + self.training)
            data_set_test = torch.load(self.root + self.testing)


        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()

        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]

        self.train_data *= 255.0
        #self.train_data = self.train_data.transpose(2, 1)
        #self.train_data = self.train_data.transpose(3, 1)

        print(self.train_data.shape)

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)

        label = label.type(torch.LongTensor)
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(self.root + self.training) and os.path.exists(self.root + self.testing)


    def load_samples(self):
        """Load sample images from dataset."""
        if self.train:
            f = self.root + self.training
        else:
            f = self.root + self.testing

        data_set = torch.load(f)

        audios = torch.Tensor([np.asarray(audio) for _, (audio, _) in enumerate(data_set)])
        labels = torch.Tensor([np.asarray([label]) for _, (_, label) in enumerate(data_set)])

        self.dataset_size = labels.shape[0]

        return audios, labels

def get_stl_10(train, dataset):

    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])

    pre_process = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    stl_10_dataset = STL_10(root=params.data_root,
                        train=train,
                        transform=pre_process,
                        download=True,
                        dataset=dataset)

    stl_10_data_loader = torch.utils.data.DataLoader(
        dataset=stl_10_dataset,
        batch_size=params.batch_size,
        shuffle=False)

    return stl_10_data_loader
