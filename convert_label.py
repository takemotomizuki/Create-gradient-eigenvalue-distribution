import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cv2
def humansize(nbytes):
    '''From https://stackoverflow.com/questions/14996453/python-libraries-to-calculate-human-readable-filesize-from-bytes'''
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])
base_path = '/root/workspace/dataset/ImageNet/data'
fns = os.listdir(base_path)
fns.sort()
fns = [
    base_path + '/' + fn
    for fn in fns
]
x_val = np.zeros((len(fns), 224, 224, 3), dtype=np.float32)
print(humansize(x_val.nbytes))
for i in range(len(fns)):
    if i %2000 == 0:
        print("%d/%d" % (i, len(fns)))
    # Load (as BGR)
    img = cv2.imread(fns[i])
    # Resize
    height, width, _ = img.shape
    new_height = height * 256 // min(img.shape[:2])
    new_width = width * 256 // min(img.shape[:2])
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    # Crop
    height, width, _ = img.shape
    startx = width//2 - (224//2)
    starty = height//2 - (224//2)
    img = img[starty:starty+224,startx:startx+224]
    assert img.shape[0] == 224 and img.shape[1] == 224, (img.shape, height, width)
    # Save (as RGB)
    x_val[i,:,:,:] = img[:,:,::-1]
np.save("data/x_val.npy", x_val)
for i in range(0,50000,10000):
    np.save("data/x_val_%d_%d.npy" % (i, i+10000), x_val[i:i+10000])
meta = scipy.io.loadmat("/root/workspace/dataset/ImageNet/ILSVRC2012_devkit_t12/data/meta.mat")
original_idx_to_synset = {}
synset_to_name = {}
for i in range(1000):
    ilsvrc2012_id = int(meta["synsets"][i,0][0][0][0])
    synset = meta["synsets"][i,0][1][0]
    name = meta["synsets"][i,0][2][0]
    original_idx_to_synset[ilsvrc2012_id] = synset
    synset_to_name[synset] = name
synset_to_keras_idx = {}
keras_idx_to_name = {}
f = open("/root/workspace/dataset/ImageNet/synset_words.txt","r")
idx = 0
for line in f:
    parts = line.split(" ")
    synset_to_keras_idx[parts[0]] = idx
    keras_idx_to_name[idx] = " ".join(parts[1:])
    idx += 1
f.close()
def convert_original_idx_to_keras_idx(idx):
    return synset_to_keras_idx[original_idx_to_synset[idx]]
f = open("/root/workspace/dataset/ImageNet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt","r")
y_val = f.read().strip().split("\n")
y_val = list(map(int, y_val))
y_val = np.array([convert_original_idx_to_keras_idx(idx) for idx in y_val])
f.close()
np.save("data/y_val.npy", y_val)