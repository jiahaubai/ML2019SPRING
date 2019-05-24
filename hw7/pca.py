from skimage import io
import numpy as np
import glob
import os
import sys

def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M
###################################################################
print('get pics')

input_image_path = sys.argv[1] #'./Aberdeen/Aberdeen/'
choose_name = sys.argv[2] #'87.jpg'
output_name = sys.argv[3] #'./reconstruct_0.jpg'

img_root = glob.glob(os.path.join(input_image_path, '*.' +'jpg'))
images = []
for idx in range(len(img_root)):
    if idx%100 == 0: print(idx)
    img = io.imread(img_root[idx])
    images.append(img.flatten())
images = np.array(images).astype('float32')

mean_face = np.mean(images,axis=0)
image_center = images - mean_face

get_img = io.imread(os.path.join(input_image_path, choose_name))
get_img = get_img.flatten().astype('float32')

print('compute SVD')
U, S, V = np.linalg.svd(image_center.T, full_matrices=False)
print("U shape",U.shape)
print("S shape",S.shape)
print("V shape",V.shape)

print('reconstruct face')
#w = np.dot(image_center[0], U[:,:5])
w = np.dot(get_img - mean_face, U[:,:5])
x = w.dot(U[:,:5].T)+mean_face
io.imsave(output_name,process(x).astype(np.uint8).reshape(600,600,3))