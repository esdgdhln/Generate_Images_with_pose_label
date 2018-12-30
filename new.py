#coding:utf-8
from  renderer import Renderer
#from utils import
import numpy as np
import cv2
from PIL import Image
from utils import precompute_projections,create_pose
from os.path import join
from model import Model3D
from os import listdir
from os.path import isfile, join
nb_bg_images = 1500           # JUST FOR TEST
#nb_bg_images = 150000      # USE THIS NUMBER TO TRAIN BB8
# Image window size


def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of veritces'
    _, R_exp, t = cv2.solvePnP(points_3D,
                               points_2D,
                               cameraMatrix,
                               distCoeffs)

    R, _ = cv2.Rodrigues(R_exp)
    Rt = np.c_[R, t]

    return Rt
#search corner



c = 3
h = 480
w = 640
background_images = np.zeros((nb_bg_images, h, w, c), dtype='uint8')
bg_path = '/home/sky/Project/BB8-J15-master/data/BG/ImageNet'
rgb_sy =  np.zeros(( h, w, c), dtype='uint8')
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '='):
    import sys
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + ' ' * (length - filledLength)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total:
        print()
def get_all_files(directory):
    files = []
    for f in listdir(directory):
        if isfile(join(directory, f)):
            files.append(join(directory, f))
        else:
            files.extend(get_all_files(join(directory, f)))
    return files

def load_bgs_to_memory():
    """
        Load background images to the memory.
    """
    bg_file_names = get_all_files(bg_path)
    print("{0} background images are found".format(len(bg_file_names)))

    print('loading background images to memory...')
    bg_nb, bg_h, bg_w = background_images.shape[:3]
    for bg_idx in range(bg_nb):
        random_bg_index = np.random.randint(0, len(bg_file_names) - 1)

        bg = cv2.imread(bg_file_names[random_bg_index])
        bg = cv2.resize(bg, (int(bg_w), int(bg_h)), interpolation=cv2.INTER_NEAREST)
        background_images[bg_idx] = bg
        printProgressBar(bg_idx, bg_nb, prefix = 'Progress:', suffix = 'Complete', length = 50)

    printProgressBar(bg_nb, bg_nb, prefix = 'Progress:', suffix = 'Complete', length = 50)
#随机的产生两个角度，就可以唯一确定viewpoint,不不不，角度的，在球面不均匀

import math

#step1：
#　　　　随机抽样产生一对均匀分布的随机数 u ，v   ；这里u，v 在[-1,1] 范围内
#　step2 ：
#　　　　计算  r^2 = u^2+v^2;
#　　　　如果 r^2 > 1 则重新抽样，直到满足   r^2 < 1  .
#  step3 ：
#　　　　计算　　
#　　　　x=2*u*sqrt(1-r2);
#　　　　y=2*v*sqrt(1-r2);
#　　　　z=1-2*r2;
ViewPoint = []

u_v = np.asarray([[0,0,1],[0.3,  0.7416  , 0.6],[1,0,0]])
ViewPoint.append(u_v[0])
ViewPoint.append(u_v[1])


class bench:
    def __init__(self):
        self.cam = np.identity(3)
        self.models = {}

Model_instance = bench()
Model_instance.cam[0, 0] = 680.0
Model_instance.cam[0, 2] = 320.0
Model_instance.cam[1, 1] = 680.0
Model_instance.cam[1, 2] = 240.0
Model_instance.cam[2, 2] = 1.0
#model file name
model_name = '2.ply'
Model_instance.models[model_name] = Model3D()
Model_instance.scale_to_meters = 0.01
#camera information
Model_instance.models[model_name].diameter = 125.64433088

Model_instance.models[model_name].load(model_name,scale = Model_instance.scale_to_meters)

views = [0.6,  0616.  , 0.6]
np.set_printoptions(precision=3, suppress=True)
import os
w = 640
h = 480
#主要存储的文件，mask文件，可以根据深度值～=0,pose真值R,T，合成的RGB图像，bb8：8个顶点值
i = 0
ren = Renderer((w, h), Model_instance.cam)
ren1 = Renderer((w, h), Model_instance.cam)
for vp,views in enumerate(ViewPoint):
    #print(views)
    views = np.asarray(views)

    pose = create_pose(views, angle_deg=0)
    zr = 0.2*np.random.rand(1,1) + 0.7
    #T = np.asarray([0, 0, zr]).T
    #Translate = np.matmul(pose[:3,:3],T)

    #pose1 = pose
    #pose1[:3, 3]  = Translate
    pose[:3, 3] = [0.0, 0., zr]  # zr = 0.5
    print('pose{}'.format(pose))
    #print('pose1{}'.format(pose1))
    ren.draw_model(Model_instance.models[model_name],pose)
    ren.draw_boundingbox(Model_instance.models[model_name], pose)
    rgb,mask = ren.finish()
    ren.clear()
    ren1.draw_model(Model_instance.models[model_name], pose)
    #ren.cal_bb8(Model_instance.models[model_name],pose)
    rgb1, mask1 = ren1.finish()
    mask1 = (mask1 > 0)
    #cv2.imshow("name",mask*255)
    #cv2.imshow("rgb", rgb)
    #cv2.waitKey(0)
    cv2.imwrite("3.png",rgb*255)
    cv2.imwrite("mask.png", mask1 * 255)

    ren1.clear()



np.set_printoptions(precision=3, suppress=True)
def get_Rt(filename):
    Rt = np.loadtxt(filename,delimiter=',', dtype='float32')[:3, :]
    return Rt
#Rt = get_Rt('0001.txt')
#print('Rt:',Rt)
Rt = pose
def get_camera_intrinsic():
    K = np.zeros((3, 4), dtype='float32')
    K[0, 0], K[0, 2] = 680, 320
    K[1, 1], K[1, 2] = 680, 240
    K[2, 2] = 1.
    return K
K = np.matmul(get_camera_intrinsic(),Rt)
Pt = np.loadtxt('J15_bb.txt', dtype='float32')[:, :]
Pts = np.zeros((8,4),dtype='float32')
Pts[:,0:3] = Pt

print('bb8:',Pts)
print(K)
pt = np.matmul(K,Pts.T)
for idx in range(0,8,1):
    pt[0,idx] = pt[0,idx]/pt[2,idx]
    pt[1,idx] = pt[1,idx]/pt[2,idx]
pt = pt[0:2,:]
print('pt:',pt)
import cv2
im = cv2.imread("3.png")
for idx in range(0,8,1):
    cv2.circle(im,(int(pt[0,idx]),int(pt[1,idx])),4,(0,255,0),-1)
cv2.imshow("projection",im)
cv2.waitKey(0)


def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d
