import argparse
from os import makedirs
from os.path import exists, join
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
import numpy as np

from general.dataset import BreastDataset
from utils.stats_tools import DataLoaderStats
from utils.os_tools import load_np_array, save_transformation


class DataModule(pl.LightningDataModule):

    @staticmethod
    def add_dataset_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # -> Datasset Args

        parser.add_argument(
            "--data_dir",
            type = str,
            default = "drive/MyDrive/vae/data",
            help = "Address of the dataset directory. [default: drive/MyDrive/vae/data]"
        )

        parser.add_argument(
            "--patch_size",
            type = int,
            default = 32,
            help = "Size of the square patches sampled from each image. [default: 32]"
        )

        parser.add_argument(
            "--n_patches_per_image",
            type = int,
            default = 100,
            help = "Number of patches that will be sampled from each image. [default: 100]"
        )

        parser.add_argument(
            "--whitespace_threshold",
            type = float,
            default = 0.82,
            help = "The threshold used for classifying a patch as mostly white space. The mean of pixel values over all channels of a patch after applying transformations is compared to this threshold. [default: 0.82]"
        )

        parser.add_argument(
            "--split_ratio",
            nargs = 3,
            type = float,
            default = [0.9, 0.05, 0.05],
            help = "The ratios we want to split the data into train/val/test. Sum of the ratios should be equal tp 1. The ratios should be seperated only with white space. [default: 0.9 0.05 0.05]"
        )

        parser.add_argument(
            "--num_dataset_workers",
            type = int,
            default = 4,
            help = "Number of processor workers used for patching images. [default: 4]"
        )
        
        parser.add_argument(
            "--STUDENT_ID",
            type = int,
            help = "Your UTA student ID."
        )

        parser.add_argument(
            "--coords_already_generated",
            action = "store_true",
            help = "If  coords.data is already generated for the dataset, it simply loads it. If not passed, coords are re-calculated. [default: If the flag is not passed --> False]"
        )

        # -> Data Module Args

        parser.add_argument(
            "--normalize_transform",
            action = argparse.BooleanOptionalAction,
            help = "If passed, DataModule will calculate or load the whole training dataset mean and std per channel and passes it to transforms."
        )

        parser.add_argument(
            "--resize_transform_size",
            type = int,
            default = None,
            help = "If provided, the every patch would be resized from patch_size to resize_transform_size. [default: None]"
        )

        parser.add_argument(
            "--num_dataloader_workers",
            type = int,
            default = 4,
            help = "Number of processor workers used for dataloaders. [default: 4]"
        )

        parser.add_argument(
            "--batch_size",
            type = int,
            default = 256,
            help = "The batch size used with all dataloaders. [default: 128]"
        )

        return parser


    def __init__(
        self,
        data_dir,
        patch_size,
        n_patches_per_image,
        whitespace_threshold,
        split_ratio,
        num_dataset_workers,
        STUDENT_ID,
        coords_already_generated,
        logging_dir,
        normalize_transform,
        resize_transform_size,
        num_dataloader_workers,
        batch_size,
        *args,
        **kwargs,
    ):
        """
        """
        super().__init__()

        self.data_dir = data_dir
        self.patch_size = patch_size
        self.n_patches_per_image = n_patches_per_image
        self.whitespace_threshold = whitespace_threshold
        self.split_ratio = split_ratio
        self.num_dataset_workers = num_dataset_workers
        self.STUDENT_ID = STUDENT_ID
        self.coords_already_generated = coords_already_generated
        self.logging_dir = logging_dir
        self.normalize_transform = normalize_transform
        self.resize_transform_size = resize_transform_size
        self.batch_size = batch_size
        self.num_dataloader_workers = num_dataloader_workers
        
        # saving hyperparameters to checkpoint
        self.save_hyperparameters()

        self.dataset_kwargs = {
            "data_dir": self.data_dir,
            "patch_size": self.patch_size,
            "n_patches_per_image": self.n_patches_per_image,
            "whitespace_threshold": self.whitespace_threshold,
            "split_ratio": self.split_ratio,
            "num_dataset_workers": self.num_dataset_workers,
            "STUDENT_ID": self.STUDENT_ID,
        }


    def prepare_data(self):
        if self.trainer.state.fn in (TrainerFn.FITTING):
            if not self.coords_already_generated:
                dataset = BreastDataset(dataset_type=None, read_coords=False, transformations=None, **self.dataset_kwargs)
            
            # Finding normalization parameters
            stats_path = join(self.data_dir, "mean.gz")
            if not exists(stats_path):
                train_dataset = BreastDataset(dataset_type="train", read_coords=True, transformations=None, **self.dataset_kwargs)
                
                # All stats should be calculated at highest stable batch_size to reduce approximation errors for mean and std
                loader = DataLoader(train_dataset, batch_size=256, num_workers=self.num_dataloader_workers)
                loader_stats = DataLoaderStats(loader, self.data_dir)

        
    def setup(self, stage=None):
        transforms_list = []
        inverse_transforms_list = []
        final_size = self.patch_size

        if self.normalize_transform:
            std = load_np_array(join(self.data_dir, "std.gz"))
            mean = load_np_array(join(self.data_dir, "mean.gz"))
            print("\n")
            print(f"mean of training set used for normalization: {mean}")
            print(f"std of training set used for normalization: {std}")
            print("\n")
            transforms_list.append(transforms.Normalize(mean=mean, std=std))
            inverse_transforms_list.insert(0, transforms.Normalize(mean=-mean, std=np.array([1, 1, 1])))
            inverse_transforms_list.insert(0, transforms.Normalize(mean=np.array([0, 0, 0]), std=1/std))

        if self.resize_transform_size is not None:
            transforms_list.append(transforms.Resize(size=self.resize_transform_size, interpolation=InterpolationMode.BILINEAR))
            inverse_transforms_list.insert(0, transforms.Resize(size=self.patch_size, interpolation=InterpolationMode.BILINEAR))
            final_size = self.resize_transform_size
        transforms_list.append(transforms.CenterCrop(final_size))

        # Composing and saving transformations and inverse transformations to file
        transformations = transforms.Compose(transforms_list)
        inverse_transformations = transforms.Compose(inverse_transforms_list)

        save_transformation(transformations, join(self.logging_dir, "trans.obj"))
        save_transformation(inverse_transformations, join(self.logging_dir, "inv_trans.obj"))

        # Creating corresponding datasets
        if stage in (None, "fit"):
            self.train_dataset = BreastDataset(dataset_type="train", read_coords=True, transformations=transformations, **self.dataset_kwargs)
            self.val_dataset = BreastDataset(dataset_type="val", read_coords=True, transformations=transformations, **self.dataset_kwargs)
        elif stage in (None, "validate"):
            self.val_dataset = BreastDataset(dataset_type="val", read_coords=True, transformations=transformations, **self.dataset_kwargs)
        elif stage in (None, "test"):
            self.test_dataset = BreastDataset(dataset_type="test", read_coords=True, transformations=transformations, **self.dataset_kwargs)

        
    def train_dataloader(self):
        train_loader = DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        num_workers=self.num_dataloader_workers,
        drop_last=True
    )
        return train_loader
        """
        #############################################################################
        ################################ PLACEHOLDER ################################
        #############################################################################

        This function should return a training Dataloader with specific inputs.
        Here is what you have to implement (~ 1 lines of code):
            - return Dataloader for self.train_dataset, batch size of self.batch_size, number of workers of self.num_dataloader_workers and drop_last to be True.
        """
        
    
    def val_dataloader(self):
        val_loader = DataLoader(
        self.val_dataset,
        batch_size=self.batch_size,
        num_workers=self.num_dataloader_workers,
        drop_last=True
    )
        return val_loader
        """
        #############################################################################
        ################################ PLACEHOLDER ################################
        #############################################################################

        This function should return a validation Dataloader with specific inputs.
        Here is what you have to implement (~ 1 lines of code):
            - return Dataloader for self.val_dataset, batch size of self.batch_size, number of workers of self.num_dataloader_workers and drop_last to be True.
        """
        pass
    
    def test_dataloader(self):
        test_loader = DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        num_workers=self.num_dataloader_workers,
        drop_last=True
    )
        return test_loader
        """
        #############################################################################
        ################################ PLACEHOLDER ################################
        #############################################################################

        This function should return a test Dataloader with specific inputs.
        Here is what you have to implement (~ 1 lines of code):
            - return Dataloader for self.test_dataset, batch size of self.batch_size, number of workers of self.num_dataloader_workers and drop_last to be True.
        """
        pass