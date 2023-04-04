import os
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class test_dataset:
    def __init__(self, image_root, gt_root):
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.tif') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.ToTensor()
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.tif'):
            name = name.split('.tif')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


if __name__ == '__main__':

    image_root = 'img_out/'  # 测试集结果路径
    gt_root = 'Medical_Datasets/Labels/'     # 测试集标签路径

    test_loader = test_dataset(image_root, gt_root)
    b = 0.0
    c = 0.0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image
        input = image[0,1,:,:]
        input = np.array(input, np.float32)

        target = np.array(gt)
        N = gt.shape
        smooth = 1


        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))


        intersection = (input_flat * target_flat)
        union = (input_flat + target_flat)

        mIoU = (intersection.sum() + smooth) / (union.sum() - intersection.sum() + smooth)

        mDICE =  (2 * intersection.sum() + smooth) / (input_flat.sum() + target_flat.sum() + smooth)

        p =  '{:.4f}'.format(mIoU)
        p = float(p)

        q =  '{:.4f}'.format(mDICE)
        q = float(q)

        b = b + p
        c = c + q
        mIoU = '{:.4f}'.format(b/test_loader.size)
        mDICE = '{:.4f}'.format(c/test_loader.size)
        # print(i, p)
        # print(i, q)
    print('mDICE:', mDICE, 'mIoU:', mIoU)

