import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class FtestDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(cache_filelist_read=True)
        parser.set_defaults(cache_filelist_write=True)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, 'nir')
        self.dir_B = os.path.join(opt.dataroot, 'rgb')
        self.dir_dark = os.path.join(opt.dataroot, 'dark')
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.dark_paths = sorted(make_dataset(self.dir_dark))
        self.opt.direction = 'AtoB'
        #assert(opt.preprocess_mode == 'resize_and_crop')

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        B_path = self.B_paths[index]
        Dark_path = self.dark_paths[index]
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        Dark = Image.open(Dark_path).convert('RGB')
        w, h = A.size
        assert(self.opt.load_size >= self.opt.crop_size)
        w_offset = random.randint(0, max(0, w - self.opt.crop_size - 1))
        h_offset = random.randint(0, max(0, h - self.opt.crop_size - 1))
        #A = A.crop((w_offset, h_offset, w_offset + self.opt.crop_size, h_offset + self.opt.crop_size))
        #B = B.crop((w_offset, h_offset, w_offset + self.opt.crop_size, h_offset + self.opt.crop_size))
        #Dark = Dark.crop((w_offset, h_offset, w_offset + self.opt.crop_size, h_offset + self.opt.crop_size))


        A = transforms.ToTensor()(A)
        Dark = transforms.ToTensor()(Dark)
        B = transforms.ToTensor()(B)

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        Dark = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(Dark)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.output_nc
        else:
            input_nc = self.opt.output_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            Dark = Dark.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'label': A, 'image': B, 'Dark': Dark,
                'label_path': A_path, 'image_path': B_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'FusionDataset'
