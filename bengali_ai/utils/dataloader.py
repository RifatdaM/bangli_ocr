import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from imageio import imread
import PIL.Image
from skimage.transform import rotate, resize
import os
import json


class BongLoader(torch.utils.data.Dataset):
    def __init__(self, output_size=(224, 224), dataset_root=None,
                 image_format='png', json_file=None, indices_file=None,
                 normalizer=None):
        self.output_size = output_size
        self.dataset_root = dataset_root
        self.image_format = image_format
        self.normalizer = normalizer
        self.image_files = []
        self.labels = []

        with open(indices_file, 'r') as f:
            data = f.read().replace('\n', '').strip().split(',')
        self.indices_o = data

        with open(json_file, 'r') as f:
            json_data = json.load(f)
        self.df = json_data

        if not self.dataset_root or not os.path.exists(self.dataset_root):
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(self.dataset_root))
        if not self.indices_o:
            raise FileNotFoundError('No dataset files found.')

        self.indices = []
        for idx in self.indices_o:
            if os.path.exists(os.path.join(self.dataset_root, 'train_images', f'{idx}.{self.image_format}')):
                self.image_files.append(os.path.join(self.dataset_root, 'train_images', f'{idx}.{self.image_format}'))
                self.indices.append(idx)

        self.length = len(self.image_files)
        # print(self.length)
        # print(len(list(self.df.keys())))
        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(self.dataset_root))

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_image(self, index):
        image = imread(self.image_files[index])
        if (image.shape[0], image.shape[1]) != self.output_size:
            image = resize(image, self.output_size, order=0, preserve_range=True).astype(image.dtype)

        image = np.array(image) // 255
        return image

    def get_label(self, index):
        df_id = self.indices[index]
        c = int(self.df[df_id]['components']['c'])
        r = int(self.df[df_id]['components']['r'])
        v = int(self.df[df_id]['components']['v'])
        grapheme = self.df[df_id]['grapheme']
        # c_head = np.zeros(8)
        # r_head = np.zeros(168)
        # v_head = np.zeros(11)
        # c_head[c] = 1
        # r_head[r] = 1
        # v_head[v] = 1
        return c, r, v, grapheme

    def __getitem__(self, index):
        img = self.get_image(index)
        c_head, r_head, v_head, grapheme = self.get_label(index)
        label_image_gray = np.zeros((img.shape[0], img.shape[1]))
        if self.normalizer:
            image, label_image_gray = self.normalizer(img, label_image_gray)
        else:
            raise NotImplementedError("Normalizer not implemented...")

        return image, c_head, r_head, v_head

    def __len__(self):
        return len(self.image_files)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    d = BongLoader(
        dataset_root='/Users/imrankabir/Desktop/research/bengali_ocr_app/bengaliai-cv19/',
        json_file='/Users/imrankabir/Desktop/research/bengali_ocr_app/bengaliai-cv19/train.json',
        indices_file='/Users/imrankabir/Desktop/research/bengali_ocr_app/bengaliai-cv19/train_indices.txt',
    )
    imgt, c_headt, r_headt, v_headt = d.__getitem__(0)

    print(c_headt, r_headt, v_headt)
    print(graphemet)
    plt.imshow(imgt)
    plt.show()
