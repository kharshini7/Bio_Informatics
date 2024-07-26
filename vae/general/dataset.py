import random
import pickle
from os import listdir
from os.path import join, exists
from datetime import datetime
from multiprocessing import Pool

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import pyvips

from utils.image_tools import vips2numpy


class BreastDataset(Dataset):
    
    def __init__(
        self,
        data_dir,
        patch_size,
        n_patches_per_image,
        whitespace_threshold,
        dataset_type,
        split_ratio,
        read_coords,
        transformations,
        num_dataset_workers,
        STUDENT_ID
    ):
        """
        """
        super().__init__()

        self.data_dir = data_dir
        self.patch_size = patch_size
        self.n_patches_per_image = n_patches_per_image
        self.whitespace_threshold = whitespace_threshold
        self.read_coords = read_coords
        self.transformations = transformations
        self.dataset_type = dataset_type
        self.split_ratio = split_ratio
        self.num_dataset_workers = num_dataset_workers

        self.patching_seed = STUDENT_ID + 1
        self.shuffle_seed = STUDENT_ID + 2
        self.test_random_seed = STUDENT_ID + 3
        self.train_val_random_seed = STUDENT_ID + 4
        
        file_names = [name for name in listdir(self.data_dir) if name.endswith(".svs")]
        file_paths = [join(self.data_dir, name) for name in file_names]

        if self.read_coords:
            if exists(join(self.data_dir, "coords.data")):
                coords = self._read_coords()
                print(f"{len(coords)} patches were loaded successfully from {self.data_dir}...")
            else:
                print(f"{join(self.data_dir, 'coords.data')} does not exist! Proceeding with calculating the coordinates...")
                coords = self._make_coords(file_paths)
        else:
            coords = self._make_coords(file_paths)

        train_coords, val_coords, test_coords = self._split(coords, self.split_ratio)

        if self.dataset_type == "train":
            self.coords = train_coords
        elif self.dataset_type == "val":
            self.coords = val_coords
        elif self.dataset_type == "test":
            self.coords = test_coords
        else:
            self.coords = coords
            

    def _make_coords(self, file_paths):
        pool = Pool(processes=self.num_dataset_workers)
        print("Multiprocessing started...")
        pool_out = pool.map(self._fetch_coords, file_paths)
        print("Multiprocessing ended successfully...")
        coords = [elem for sublist in pool_out for elem in sublist]
        random.seed(self.shuffle_seed)
        random.shuffle(coords)
        self._write_coords(coords)
        print(f"{len(coords)} patches were created and written successfully in {self.data_dir}...")
        return coords
    

    def _fetch_coords(self, file_path):
        print(f"{file_path} loaded to be patched...\n", flush=True)
        image = self._load_file(file_path)
        patches = self._patching(image, seed=self.patching_seed)
        paths = [file_path] * len(patches)
        return list(zip(paths, patches))


    def _load_file(self, file_path):
        image = pyvips.Image.new_from_file(str(file_path))
        return image
    

    def _patching(self, image, seed):
        if seed is not None:
            random.seed(seed)
        
        # Making sure not to spend more than 5 mins per image to find the required number of patches
        start_time = datetime.now()
        spent_time = datetime.now() - start_time

        count = 0
        coords = []
        while count < self.n_patches_per_image and spent_time.total_seconds() < 500:
            # [4, x, y] -> many [4, 512, 512]
            rand_i = random.randint(0, image.width - self.patch_size)
            rand_j = random.randint(0, image.height - self.patch_size)
            temp = self._image_to_tensor(image, rand_i, rand_j)
            if self._filter_whitespace(temp, threshold=self.whitespace_threshold):
                if self._get_intersections(rand_j, rand_i, coords):
                    coords.append((rand_i, rand_j))
                    count += 1
            spent_time = datetime.now() - start_time
        return coords


    def _image_to_tensor(self, image, x, y):
        t = image.crop(x, y, self.patch_size, self.patch_size)
        t_np = vips2numpy(t)
        trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
        out_t = trans(t_np)
        out_t = out_t[:3, :, :]
        return out_t

    
    def _filter_whitespace(self, tensor_3d, threshold):
        r = np.mean(np.array(tensor_3d[0]))
        g = np.mean(np.array(tensor_3d[1]))
        b = np.mean(np.array(tensor_3d[2]))
        channel_avg = np.mean(np.array([r, g, b]))
        if channel_avg < threshold:
            return True
        else:
            return False


    def _get_intersections(self, x, y, coords):
        if len(coords) == 0:
            return True
        else:
            ml = set(map(lambda b: self._get_intersection(b[0], b[1], x, y), coords))
            if True in ml:
                return False
            else: 
                return True


    def _get_intersection(self, a_x, a_y, b_x, b_y):
        # Tensors are row major
        if abs(a_x - b_x) < self.patch_size and abs(a_y - b_y) < self.patch_size:
            return True
        else:
            return False


    def _read_coords(self):
        with open(join(self.data_dir, "coords.data"), 'rb') as file_handle:
                coords = pickle.load(file_handle)
                file_handle.close()
        return coords

    
    def _write_coords(self, coords):
        with open(join(self.data_dir, "coords.data"), 'wb') as file_handle:
            pickle.dump(coords, file_handle)
            file_handle.close()


    def _split(self, coords, split_ratio):
        n = len(coords)
        train_size = int(split_ratio[0] * n)
        val_size = int(split_ratio[1] * n)
        test_size = n - train_size - val_size

        random.seed(self.test_random_seed) 
        random.shuffle(coords)
        test_coords = coords[:test_size]
        train_coords = coords[test_size:]
        
        random.seed(self.train_val_random_seed)
        random.shuffle(train_coords)
        val_coords = train_coords[train_size:]
        train_coords = train_coords[:train_size]

        print(f"There are {len(train_coords)} patches in training set...")
        print(f"There are {len(val_coords)} patches in validation set...")
        print(f"There are {len(test_coords)} patches in test set...")

        return train_coords, val_coords, test_coords


    def __getitem__(self, index):
            # the items path and coordinates are retrieved
        file_path, coord = self.coords[index]
        
        # Loading the image file
        image_h = self._load_file(file_path)
        
        # image patch to a tensor
        out = self._image_to_tensor(image_h, coord[0], coord[1])
        
        # Applying if there are any transformations
        if self.transformations:
            out = self.transformations(out)
        
        # Return the tensor and its size
        return out, out.size()
        """
        #############################################################################
        ################################ PLACEHOLDER ################################
        #############################################################################

        This function should return the index'th item in our dataset and its size (2 outputs).
        Here is what you have to implement (~ 6 lines of code):
            - Read the index'th item in self.coords. it is a tuple of index'th item file_path and coord.
            - Using self._load_file() function load the index'th item image
            - Using self._image_to_tensor() function get the actual pixel values for that specific patch and store it in a variable called "out".
            - If self.transformations is not empty, you should apply it on "out".
            - Return "out" and the size of "out" as outputs
        """
        


    def __len__(self):
        return len(self.coords)
        """
        #############################################################################
        ################################ PLACEHOLDER ################################
        #############################################################################
        
        This function should return the size of the dataset (i.e. number of all patches generated).
        Here is what you have to implement (~ 1 lines of code):
            - return the length of "self.coords".s
        """
        