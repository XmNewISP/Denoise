#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

from utils import load_hdr_as_tensor

import os
from sys import platform
import numpy as np
import random
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw
import OpenEXR

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def load_dataset(root_dir, redux, params, shuffled=False, single=False):
    """Loads dataset and returns corresponding data loader."""

    # Create Torch dataset
    noise = (params.noise_type, params.noise_param)

    # Instantiate appropriate dataset class
    if params.noise_type == 'mc':
        dataset = MonteCarloDataset(root_dir, redux, params.crop_size,
            clean_targets=params.clean_targets)
    elif params.noise_type in ['gaussian', 'poisson','havenoise']:
        dataset = NoisyDataset(root_dir, redux, params.crop_size,
            clean_targets=params.clean_targets, noise_dist=noise, seed=params.seed)
    elif params.noise_type == 'realnoise':
        dataset = MyNoisyDataset(root_dir, redux, params.crop_size,
            clean_targets=params.clean_targets)
    else:
        assert None,f'Error: Unknown noise_type: {self.noise_type})'

    # Use batch size of 1, if requested (e.g. test set)
    if single:
        return DataLoader(dataset, batch_size=1, shuffle=shuffled,drop_last=True)
    else:
        return DataLoader(dataset, batch_size=params.batch_size, shuffle=shuffled,num_workers=params.workers_num,drop_last=True)


class AbstractDataset(Dataset):
    """Abstract dataset class for Noise2Noise."""

    def __init__(self, root_dir, redux=0, crop_size=128, clean_targets=False):
        """Initializes abstract dataset."""

        super(AbstractDataset, self).__init__()

        self.imgs = []
        self.root_dir = root_dir
        self.redux = redux
        self.crop_size = crop_size
        self.clean_targets = clean_targets

    def _random_crop(self, img_list):
        """Performs random square crop of fixed size.
        Works with list so that all items get the same cropped window (e.g. for buffers).
        """

        w, h = img_list[0].size
        assert w >= self.crop_size and h >= self.crop_size, \
            f'Error: Crop size: {self.crop_size}, Image size: ({w}, {h})'
        cropped_imgs = []
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        for img in img_list:
            # Resize if dimensions are too small
            if min(w, h) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))

            # Random crop
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))

        return cropped_imgs


    def __getitem__(self, index):
        """Retrieves image from data folder."""

        raise NotImplementedError('Abstract method not implemented!')


    def __len__(self):
        """Returns length of dataset."""
        print("Dataset--------->len=",len(self.imgs))
        return len(self.imgs)


class NoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""

    def __init__(self, root_dir, redux, crop_size, clean_targets=False,
        noise_dist=('gaussian', 50.), seed=None):
        """Initializes noisy image dataset."""

        super(NoisyDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        self.imgs = os.listdir(root_dir)
        self.imgs.remove("denoised") if "denoised" in self.imgs else None
        if redux:
            self.imgs = self.imgs[:redux]

        # Noise parameters (max std for Gaussian, lambda for Poisson, nb of artifacts for text)
        self.noise_type = noise_dist[0]
        self.noise_param = noise_dist[1]
        self.seed = seed
        if self.seed:
            np.random.seed(self.seed)


    def _add_noise(self, img):
        """Adds Gaussian or Poisson noise to image."""

        w, h = img.size
        c = len(img.getbands())

        # Poisson distribution
        # It is unclear how the paper handles this. Poisson noise is not additive,
        # it is data dependent, meaning that adding sampled valued from a Poisson
        # will change the image intensity...
        if self.noise_type == 'poisson':
            noise = np.random.poisson(img)
            noise_img = img + noise
            noise_img = 255 * (noise_img / np.amax(noise_img))

        # Normal distribution (default)
        else:
            if self.seed:
                std = self.noise_param
            else:
                std = np.random.uniform(0, self.noise_param)
            noise = np.random.normal(0, std, (h, w, c))

            # Add noise and clip
            noise_img = np.array(img) + noise

        noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noise_img)


    def _add_text_overlay(self, img):
        """Adds text overlay to images."""

        assert self.noise_param < 1, 'Text parameter is an occupancy probability'

        w, h = img.size
        c = len(img.getbands())

        # Choose font and get ready to draw
        if platform == 'linux':
            serif = '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf'
        else:
            serif = 'Times New Roman.ttf'
        text_img = img.copy()
        text_draw = ImageDraw.Draw(text_img)

        # Text binary mask to compute occupancy efficiently
        w, h = img.size
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        # Random occupancy in range [0, p]
        if self.seed:
            random.seed(self.seed)
            max_occupancy = self.noise_param
        else:
            max_occupancy = np.random.uniform(0, self.noise_param)
        def get_occupancy(x):
            y = np.array(x, dtype=np.uint8)
            return np.sum(y) / y.size

        # Add text overlay by choosing random text, length, color and position
        while 1:
            font = ImageFont.truetype(serif, np.random.randint(16, 21))
            length = np.random.randint(10, 25)
            chars = ''.join(random.choice(ascii_letters) for i in range(length))
            color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)

            # Update mask and check occupancy
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break

        return text_img


    def _corrupt(self, img):
        """Corrupts images (Gaussian, Poisson, or text overlay)."""

        if self.noise_type in ['gaussian', 'poisson']:
            return self._add_noise(img)
        elif self.noise_type == 'text':
            return self._add_text_overlay(img)
        elif self.noise_type == 'havenoise':  #已经有噪声了，不用再加噪声
            return img
        else:
            raise ValueError('Invalid noise type: {}'.format(self.noise_type))


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Load PIL image
        img_path = os.path.join(self.root_dir, self.imgs[index])
        img =  Image.open(img_path).convert('RGB')
        # Random square crop
        if self.crop_size != 0:
            img = self._random_crop([img])[0]

        # Corrupt source image
        #tmp = self._corrupt(img)
        source = tvF.to_tensor(self._corrupt(img))

        # Corrupt target image, but not when clean targets are requested
        if self.clean_targets:
            target = tvF.to_tensor(img)
        else:
            target = tvF.to_tensor(self._corrupt(img))
        return source, target


class MonteCarloDataset(AbstractDataset):
    """Class for dealing with Monte Carlo rendered images."""

    def __init__(self, root_dir, redux, crop_size,
        hdr_buffers=False, hdr_targets=True, clean_targets=False):
        """Initializes Monte Carlo image dataset."""

        super(MonteCarloDataset, self).__init__(root_dir, redux, crop_size, clean_targets)

        # Rendered images directories
        self.root_dir = root_dir
        self.imgs = os.listdir(os.path.join(root_dir, 'render'))
        self.albedos = os.listdir(os.path.join(root_dir, 'albedo'))
        self.normals = os.listdir(os.path.join(root_dir, 'normal'))

        if redux:
            self.imgs = self.imgs[:redux]
            self.albedos = self.albedos[:redux]
            self.normals = self.normals[:redux]

        # Read reference image (converged target)
        ref_path = os.path.join(root_dir, 'reference.png')
        self.reference = Image.open(ref_path).convert('RGB')

        # High dynamic range images
        self.hdr_buffers = hdr_buffers
        self.hdr_targets = hdr_targets


    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""

        # Use converged image, if requested
        if self.clean_targets:
            target = self.reference
        else:
            target_fname = self.imgs[index].replace('render', 'target')
            file_ext = '.exr' if self.hdr_targets else '.png'
            target_fname = os.path.splitext(target_fname)[0] + file_ext
            target_path = os.path.join(self.root_dir, 'target', target_fname)
            if self.hdr_targets:
                target = tvF.to_pil_image(load_hdr_as_tensor(target_path))
            else:
                target = Image.open(target_path).convert('RGB')

        # Get buffers
        render_path = os.path.join(self.root_dir, 'render', self.imgs[index])
        albedo_path = os.path.join(self.root_dir, 'albedo', self.albedos[index])
        normal_path =  os.path.join(self.root_dir, 'normal', self.normals[index])

        if self.hdr_buffers:
            render = tvF.to_pil_image(load_hdr_as_tensor(render_path))
            albedo = tvF.to_pil_image(load_hdr_as_tensor(albedo_path))
            normal = tvF.to_pil_image(load_hdr_as_tensor(normal_path))
        else:
            render = Image.open(render_path).convert('RGB')
            albedo = Image.open(albedo_path).convert('RGB')
            normal = Image.open(normal_path).convert('RGB')

        # Crop
        if self.crop_size != 0:
            buffers = [render, albedo, normal, target]
            buffers = [tvF.to_tensor(b) for b in self._random_crop(buffers)]

        # Stack buffers to create input volume
        source = torch.cat(buffers[:3], dim=0)
        target = buffers[3]

        return source, target

#geyijun@2020-03-13
#用真实的图像训练;同时支持从子目录中去取图。
class MyNoisyDataset(AbstractDataset):
    """Class for injecting random noise into dataset."""
    """A dataset for loading filelist as.
    like::
        train/case1/1.jpg
        train/case1/2.jpg
        train/case1/N.jpg
        train/case2/1.jpg
        train/case2/2.jpg
        train/case2/N.jpg
    """
    def __init__(self, root_dir, redux, crop_size, clean_targets=False):
        """Initializes noisy image dataset."""
        super(MyNoisyDataset, self).__init__(root_dir, redux, crop_size, clean_targets)
        #列出所有的场景目录的文件列表
        self.imgs = []
        self.case_indexmap = {}    #casename->(begin,end)
        caselist = os.listdir(root_dir)
        begin_index = 0
        for case in caselist:
            namelist = os.listdir(os.path.join(root_dir,case))
            filelist=[os.path.join(root_dir,case,name) for name in namelist if (name != "groundtruth.jpg")]
            self.imgs.extend(filelist)
            self.case_indexmap[os.path.join(root_dir,case)] = (begin_index,begin_index+len(filelist))
            begin_index += len(filelist)

        if redux:
            self.imgs = self.imgs[:redux]
        print("MyNoisyDataset,imgs=",len(self.imgs))
        print("MyNoisyDataset,case_indexmap=",self.case_indexmap)
        
    def __getitem__(self, index):
        """Retrieves image from folder and corrupts it."""
        # Load PIL image
        img_path = self.imgs[index]
        img =  Image.open(img_path).convert('RGB')
        if self.clean_targets:
            target_path = os.path.join(os.path.split(img_path)[0],"groundtruth.jpg")
        else:
            begin_index,end_index = self.case_indexmap[os.path.split(img_path)[0]]
            random_index = np.random.randint(begin_index,end_index)
            if(random_index == index):
                random_index = np.random.randint(begin_index,end_index)
            target_path = self.imgs[random_index]
            assert (os.path.split(img_path)[0] == os.path.split(target_path)[0])
        target = Image.open(target_path).convert('RGB')
        # Random square crop
        if self.crop_size != 0:
            crops_imgs = self._random_crop([img,target])
            #crops_imgs[0].save('crop0.png')
            #crops_imgs[1].save('crop1.png')
            #print("---test---")
            #exit()
        # source image
        source = tvF.to_tensor(crops_imgs[0])
        target = tvF.to_tensor(crops_imgs[1])
        return source, target

