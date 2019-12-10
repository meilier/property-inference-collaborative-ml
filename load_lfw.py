from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
#from scipy.misc import  imresize

from PIL import Image
from imageio import imread
import pandas as pd
import numpy as np
import warnings

import os


DATA_DIR = './data/'
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)


LFW_DIR = DATA_DIR + '/lfw_home/lfw_funneled/'


def download_lfw_raw():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fetch_lfw_people(color=True, data_home=DATA_DIR)


def save_lfw(slice_=(slice(70, 195), slice(78, 172)), resize=0.5):
    attr = pd.read_csv('./data/lfw_attributes.txt', delimiter='\t')
    names = np.asarray(attr['person'])
    img_num = np.asarray(attr['imagenum'])

    default_slice = (slice(0, 250), slice(0, 250))
    slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))
    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)

    imgs = np.zeros((len(names), h, w, 3), dtype=np.uint8)
    i = 0
    for name, num in zip(names, img_num):
        name = name.replace(' ', '_')
        img_path = os.path.join(LFW_DIR, name, '{}_{}.jpg'.format(name, str(num).zfill(4)))
        img = imread(img_path)[slice_]
        #img = imresize(img, resize)
        img = np.array(Image.fromarray(img).resize((w, h)))
        imgs[i] = img
        i += 1

    np.savez(DATA_DIR + 'lfw_images.npz', imgs)


def load_lfw_binary_attr(attr_type='gender'):
    attr = pd.read_csv(DATA_DIR + 'lfw_attributes.txt', delimiter='\t')
    if attr_type == 'gender':
        gender = np.asarray(attr['Male'])
        # the column of Male > 0 set 1, <=0 set 0
        gender = np.sign(gender)
        gender[gender == -1] = 0
        # print len(gender), np.mean(gender == 1), np.mean(gender == 0)
        return dict(zip(range(len(gender)), gender))
    elif attr_type == 'smile':
        smile = np.asarray(attr['Smiling'])
        smile = np.sign(smile)
        smile[smile == -1] = 0
        return dict(zip(range(len(smile)), smile))
    else:
        raise ValueError(attr_type)


def load_lfw_multi_attr(attr_type='race', thresh=-0.1):
    attr = pd.read_csv(DATA_DIR + 'lfw_attributes.txt', delimiter='\t')

    if attr_type == 'race':
        # extract 'Asian', 'White', 'Black' from lfw_attributes
        attr = np.asarray(attr[MULTI_ATTRS['race']])
    elif attr_type == 'glasses':
        attr = np.asarray(attr[MULTI_ATTRS['glasses']])
    elif attr_type == 'age':
        attr = np.asarray(attr[MULTI_ATTRS['age']])
    elif attr_type == 'hair':
        attr = np.asarray(attr[MULTI_ATTRS['hair']])
    else:
        raise ValueError(attr_type)

    indices = []
    labels = []
    for i, a in enumerate(attr):
        if np.max(a) < thresh:  # score too low for an attribute
            continue
        indices.append(i)
        # get indice for the max value in a, 删除不容易判断的数据 属性< -0.1
        labels.append(np.argmax(a))

    return dict(zip(indices, labels))


BINARY_ATTRS = {'gender': ['Female', 'Male'],
                'smile': ['Not Smiling', 'Smiling']}

MULTI_ATTRS = {'race': ['Asian', 'White', 'Black'],
               'glasses': ['Eyeglasses', 'Sunglasses', 'No Eyewear'],
               'age': ['Baby', 'Child', 'Youth', 'Middle Aged', 'Senior'],
               'hair': ['Black Hair', 'Blond Hair', 'Brown Hair', 'Bald']}


def load_lfw_attr(attr='gender'):
    #load_lfw_binary_attr to load gender, load_lfw_multi_attr to load race
    return load_lfw_binary_attr(attr) if attr in BINARY_ATTRS else load_lfw_multi_attr(attr)


def load_lfw_with_attrs(attr1='gender', attr2=None):
    if not os.path.exists(DATA_DIR + 'lfw_images.npz'):
        download_lfw_raw()
        save_lfw()

    with np.load(DATA_DIR + 'lfw_images.npz') as f:
        imgs = f['arr_0'].transpose(0, 3, 1, 2)

    #extract gender imformation from lfw_attributes.txt as a dict, male 1,female 0
    index_label_1 = load_lfw_attr(attr1)
    if attr2 is None:
        indices = np.sort(index_label_1.keys())
        imgs = imgs[indices] / np.float32(255.0)
        labels = np.asarray([index_label_1[i] for i in indices], dtype=np.int32)
        return imgs, labels
    #
    index_label_2 = load_lfw_attr(attr2)
    # 提取出来最容易判读的数据集的公共目录，common_indices 的size 即是11644
    common_indices = np.intersect1d(list(index_label_1.keys()), list(index_label_2.keys()))
    # 提取出图像，RBG在
    imgs = imgs[common_indices] / np.float32(255.0)
    # 返回实验需要使用的数据集的labels1性别信息，labels2 :'Asian', 'White', 'Black'信息
    labels1 = np.asarray([index_label_1[i] for i in common_indices], dtype=np.int32)
    labels2 = np.asarray([index_label_2[i] for i in common_indices], dtype=np.int32)

    return imgs, labels1, labels2
