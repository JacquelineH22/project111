'''
Copied from https://github.com/vqdang/hover_net.
Modified for different datasets used.

Patch extraction script.
'''

import re
import glob
import os
import pathlib
import shutil
import math

import cv2
import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt
import tqdm


class __AbstractDataset(object):
    '''Abstract class for interface of subsequent classes.
    Main idea is to encapsulate how each dataset should parse
    their images and annotations.

    '''

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError


class __CPM17(__AbstractDataset):
    '''Defines the CPM 2017 dataset as originally introduced in:

    Vu, Quoc Dang, Simon Graham, Tahsin Kurc, Minh Nguyen Nhat To, Muhammad Shaban,
    Talha Qaiser, Navid Alemi Koohbanani et al. 'Methods for segmentation and classification
    of digital microscopy tissue images.' Frontiers in bioengineering and biotechnology 7 (2019).

    '''

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, 'Not support'
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)['inst_map']
        ann_inst = ann_inst.astype('int32')
        ann = np.expand_dims(ann_inst, -1)
        return ann


class __Kumar(__AbstractDataset):
    '''Defines the Kumar dataset as originally introduced in:

    Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane,
    and Amit Sethi. 'A dataset and a technique for generalized nuclear segmentation for
    computational pathology.' IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560.

    '''

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        assert not with_type, 'Not support'
        ann_inst = sio.loadmat(path)['inst_map']
        ann_inst = ann_inst.astype('int32')
        ann = np.expand_dims(ann_inst, -1)
        return ann


class __CoNSeP(__AbstractDataset):
    '''Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak,
    and Nasir Rajpoot. 'Hover-Net: Simultaneous segmentation and classification of nuclei in
    multi-tissue histology images.' Medical Image Analysis 58 (2019): 101563

    '''

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=True):
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)['inst_map']
        if with_type:
            ann_type = sio.loadmat(path)['type_map']

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype('int32')
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype('int32')

        return ann


class __PanNuke(__AbstractDataset):
    def load_img(self, path):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        return img

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        ann_data = sio.loadmat(path)
        ann_inst = ann_data['inst_map']

        if with_type:
            ann_type = ann_data['type_map']

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype('int32')
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype('int32')

        return ann


def get_dataset(name):
    '''Return a pre-defined dataset object associated with `name`.'''
    name_dict = {
        'cpm17': lambda: __CPM17(),
        'kumar': lambda: __Kumar(),
        'consep': lambda: __CoNSeP(),
        'pannuke': lambda:__PanNuke(),
    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, 'Unknown dataset `%s`' % name


def rm_n_mkdir(dir_path):
    '''Remove and make directory.'''
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def cropping_center(x, crop_shape, batch=False):
    '''Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array

    Returns:
        x: cropped array

    '''
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0: h0 + crop_shape[0], w0: w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0: h0 + crop_shape[0], w0: w0 + crop_shape[1]]
    return x


class PatchExtractor(object):
    '''Extractor to generate patches with or without padding.
    Turn on debug mode to see how it is done.

    Args:
        x         : input image, should be of shape HWC
        win_size  : a tuple of (h, w)
        step_size : a tuple of (h, w)
        debug     : flag to see how it is done
    Return:
        a list of sub patches, each patch has dtype same as x

    Examples:
        >>> xtractor = PatchExtractor((450, 450), (120, 120))
        >>> img = np.full([1200, 1200, 3], 255, np.uint8)
        >>> patches = xtractor.extract(img, 'mirror')

    '''

    def __init__(self, win_size, step_size, debug=False):

        self.patch_type = 'mirror'
        self.win_size = win_size
        self.step_size = step_size
        self.debug = debug
        self.counter = 0

    def __get_patch(self, x, ptx):
        pty = (ptx[0] + self.win_size[0], ptx[1] + self.win_size[1])
        win = x[ptx[0]: pty[0], ptx[1]: pty[1]]
        assert (
                win.shape[0] == self.win_size[0] and win.shape[1] == self.win_size[1]
        ), '[BUG] Incorrect Patch Size {0}'.format(win.shape)
        if self.debug:
            if self.patch_type == 'mirror':
                cen = cropping_center(win, self.step_size)
                cen = cen[..., self.counter % 3]
                cen.fill(150)
            cv2.rectangle(x, ptx, pty, (255, 0, 0), 2)
            plt.imshow(x)
            plt.show(block=False)
            plt.pause(1)
            plt.close()
            self.counter += 1
        return win

    def __extract_valid(self, x):
        '''Extracted patches without padding, only work in case win_size > step_size.

        Note: to deal with the remaining portions which are at the boundary a.k.a
        those which do not fit when slide left->right, top->bottom), we flip
        the sliding direction then extract 1 patch starting from right / bottom edge.
        There will be 1 additional patch extracted at the bottom-right corner.

        Args:
            x         : input image, should be of shape HWC
            win_size  : a tuple of (h, w)
            step_size : a tuple of (h, w)
        Return:
            a list of sub patches, each patch is same dtype as x

        '''
        im_h = x.shape[0]
        im_w = x.shape[1]

        def extract_infos(length, win_size, step_size):
            flag = (length - win_size) % step_size != 0
            last_step = math.floor((length - win_size) / step_size)
            last_step = (last_step + 1) * step_size
            return flag, last_step

        h_flag, h_last = extract_infos(im_h, self.win_size[0], self.step_size[0])
        w_flag, w_last = extract_infos(im_w, self.win_size[1], self.step_size[1])

        sub_patches = []
        #### Deal with valid block
        for row in range(0, h_last, self.step_size[0]):
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        #### Deal with edge case
        if h_flag:
            row = im_h - self.win_size[0]
            for col in range(0, w_last, self.step_size[1]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        if w_flag:
            col = im_w - self.win_size[1]
            for row in range(0, h_last, self.step_size[0]):
                win = self.__get_patch(x, (row, col))
                sub_patches.append(win)
        if h_flag and w_flag:
            ptx = (im_h - self.win_size[0], im_w - self.win_size[1])
            win = self.__get_patch(x, ptx)
            sub_patches.append(win)
        return sub_patches

    def __extract_mirror(self, x):
        '''Extracted patches with mirror padding the boundary such that the
        central region of each patch is always within the orginal (non-padded)
        image while all patches' central region cover the whole orginal image.

        Args:
            x         : input image, should be of shape HWC
            win_size  : a tuple of (h, w)
            step_size : a tuple of (h, w)
        Return:
            a list of sub patches, each patch is same dtype as x

        '''
        diff_h = self.win_size[0] - self.step_size[0]
        padt = diff_h // 2
        padb = diff_h - padt

        diff_w = self.win_size[1] - self.step_size[1]
        padl = diff_w // 2
        padr = diff_w - padl

        pad_type = 'constant' if self.debug else 'reflect'
        x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), pad_type)
        sub_patches = self.__extract_valid(x)
        return sub_patches

    def extract(self, x, patch_type):
        patch_type = patch_type.lower()
        self.patch_type = patch_type
        if patch_type == 'valid':
            return self.__extract_valid(x)
        elif patch_type == 'mirror':
            return self.__extract_mirror(x)
        else:
            assert False, 'Unknown Patch Type [%s]' % patch_type
        return


# -------------------------------------------------------------------------------------
if __name__ == '__main__':

    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = True

    win_size = [384, 384]
    step_size = [128, 128]
    extract_type = 'mirror'  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use cpm17, kumar, consep, monusac or pannuke.
    # This is  used to get the specific dataset img and ann loading scheme
    dataset_name = 'pannuke'
    save_root = '/data1/hounaiqiao/SpatioCell/datasets/type/pannuke/patches/'
    
    # a dictionary to specify where the dataset path should be
    dataset_info = {
        'fold1': {
            'img': ('.png', '/data1/hounaiqiao/yy/NucleiSeg/benchmark/data/pannuke/fold1/image/'),
            'ann': ('.mat', '/data1/hounaiqiao/yy/NucleiSeg/benchmark/data/pannuke/fold1/label/'),
        },
        'fold2': {
            'img': ('.png', '/data1/hounaiqiao/yy/NucleiSeg/benchmark/data/pannuke/fold2/image/'),
            'ann': ('.mat', '/data1/hounaiqiao/yy/NucleiSeg/benchmark/data/pannuke/fold2/label/'),
        },
        'fold3': {
            'img': ('.png', '/data1/hounaiqiao/yy/NucleiSeg/benchmark/data/pannuke/fold3/image/'),
            'ann': ('.mat', '/data1/hounaiqiao/yy/NucleiSeg/benchmark/data/pannuke/fold3/label/'),
        },
    }

    patterning = lambda x: re.sub('([\[\]])', '[\\1]', x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc['img']
        ann_ext, ann_dir = split_desc['ann']

        out_dir = '%s/%s/%dx%d_%dx%d/' % (
            save_root,
            split_name,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
        )
        file_list = glob.glob(patterning('%s/*%s' % (ann_dir, ann_ext)))
        file_list.sort()  # ensure same ordering across platform

        rm_n_mkdir(out_dir)

        pbar_format = 'Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]'
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = pathlib.Path(file_path).stem

            img = parser.load_img('%s/%s%s' % (img_dir, base_name, img_ext))
            ann = parser.load_ann(
                '%s/%s%s' % (ann_dir, base_name, ann_ext), type_classification
            )

            # *
            img = np.concatenate([img, ann], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = 'Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                np.save('{0}/{1}_{2:03d}.npy'.format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()
            # *

            pbarx.update()
        pbarx.close()

    print('Done!')
