"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as pltimg



def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def visualize_unpaired(path, kind="train"):
    f, axarr = plt.subplots(2, 4, figsize=(20,10))
    f.suptitle("Random Images From Both Sets", fontsize=30)
    if os.path.exists(path+"/{}A".format(kind)):
        files = os.listdir(path+"/{}A".format(kind))
        for i in range(2):
            for j in range(2):     
                file = random.choice(files)
                axarr[i, j].imshow(pltimg.imread(path+"/{}A/".format(kind)+file)) 
    if os.path.exists(path+"/{}B".format(kind)):
        files = os.listdir(path+"/{}B".format(kind))
        for i in range(2):
            for j in range(2,4): 
                file = random.choice(files)
                axarr[i, j].imshow(pltimg.imread(path+"/{}B/".format(kind)+file)) 
    plt.show()

def resolution_distribution(path, kind="train"):
    a_widths = []
    a_heights = []
    if os.path.exists(path+"/{}A".format(kind)):
        files = os.listdir(path+"/{}A".format(kind))
        for file in files:
            im = Image.open(path+"/{}A/".format(kind)+file)
            width, height = im.size
            a_widths.append(width)
            a_heights.append(height)
    b_widths = []
    b_heights = []
    if os.path.exists(path+"/{}B".format(kind)):
        files = os.listdir(path+"/{}B".format(kind))
        for file in files:
            im = Image.open(path+"/{}B/".format(kind)+file)
            width, height = im.size
            b_widths.append(width)
            b_heights.append(height)
            if height<256:
                print(file)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    if a_heights and b_heights:
        sizes = [a_heights, b_heights, a_widths, b_widths]
    elif a_heights:
        sizes = [a_heights, a_widths]
    elif b_heights:
        sizes = [b_heights, b_widths]
    bp = ax.violinplot(sizes)
    plt.show()
    
    a_heights_arr = np.array(a_heights)
    a_widths_arr = np.array(a_widths)
    b_heights_arr = np.array(b_heights)
    b_widths_arr = np.array(b_widths)
    if(a_heights):
        print("Set A Heights: Min: {}, Mean: {}, Max: {}".format(a_heights_arr.min(), a_heights_arr.mean(), a_heights_arr.max()))
        print("Set A Widths: Min: {}, Mean: {}, Max: {}".format(a_widths_arr.min(), a_widths_arr.mean(), a_widths_arr.max()))
    if(b_heights):
        print("Set B Heights: Min: {}, Mean: {}, Max: {}".format(b_heights_arr.min(), b_heights_arr.mean(), b_heights_arr.max()))
        print("Set B Widths: Min: {}, Mean: {}, Max: {}".format(b_widths_arr.min(), b_widths_arr.mean(), b_widths_arr.max()))
    
def scale_images(path, factor):
    files = os.listdir(path)
    for file in files:
        filepath = os.path.join(path,file)
        img = Image.open(filepath)
        width, height = img.size
        img.thumbnail((width*factor, height*factor))
        img.save(filepath)
        
    
        
    
    
    
    
