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
c = 3
h = 480
w = 640
background_images = np.zeros((nb_bg_images, h, w, c), dtype='uint8')
bg_path = '/home/sky/Project/BB8-J15-master/data/BG/ImageNet'

def get_all_files(directory):
    files = []

    for f in listdir(directory):
        if isfile(join(directory, f)):
            files.append(join(directory, f))
        else:
            files.extend(get_all_files(join(directory, f)))
    return files

def load_bgs_to_memory():
    bg_file_names = get_all_files(bg_path)
    print("{0} background images are found".format(len(bg_file_names)))

    print('loading background images to memory...')
    bg_nb, bg_h, bg_w = background_images.shape[:3]
    for bg_idx in range(bg_nb):
        random_bg_index = random.randint(0, len(bg_file_names) - 1)

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
u_v = (2*np.random.rand(1000,3)-1)
#u_v = np.asarray([[0.6,  0.616  , 0.6],[1,  0, 0],[0,1,0],[0,0,1]])
#ViewPoint.append(u_v[0])
#ViewPoint.append(u_v[1])
#ViewPoint.append(u_v[2])
#ViewPoint.append(u_v[3])

for idx in range(0,1000,1):
    u = u_v[idx][0]
    v = u_v[idx][1]
    r2 = u*u + v*v
    if(r2 <1):
        x = 2*u*math.sqrt(1-r2)
        y = 2*v*math.sqrt(1-r2)
        z = 1 - 2*r2
        if(z >0.10 and y>0.40):
            viewpoint = [x,y,z]
            ViewPoint.append(viewpoint)
            #添加



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
model_name = '1.ply'
Model_instance.models[model_name] = Model3D()
Model_instance.scale_to_meters = 0.01
#camera information
Model_instance.models[model_name].diameter = 125.64433088

Model_instance.models[model_name].load(model_name,scale = Model_instance.scale_to_meters)

#views = [0.6,  0616.  , 0.6]

import os
w = 640
h = 480
#主要存储的文件，mask文件，可以根据深度值～=0,pose真值R,T，合成的RGB图像，bb8：8个顶点值
i = 0
ren = Renderer((w, h), Model_instance.cam)
for vp,views in enumerate(ViewPoint):
    #print(views)
    views = np.asarray(views)
    for angle_deg in xrange(-30,30,10):

        pose = create_pose(views, angle_deg=angle_deg)
        zr = 2*np.random.rand(1,1) + 0.7
        pose[:3, 3] = [0.0, 0., zr]  # zr = 0.5
        pose_path = join(os.path.dirname(__file__), 'pose')
        content = pose_path + "/{:04d}".format(i)+ ".txt"
        with open(content,'ab') as abc:
            np.savetxt(abc,pose,delimiter=",")
            abc.write("\n")
        ren.draw_model(Model_instance.models[model_name],pose)
        rgb,mask = ren.finish()

        rgb_path = join(os.path.dirname(__file__),'Image')
        mask_path = join(os.path.dirname(__file__), 'mask')
        if(os.path.exists(rgb_path)):
            rgb_name = rgb_path +"/"+"Render_{:04d}.jpg".format(i)
        else:
            os.mkdir(rgb_path)
            rgb_name = rgb_path + "/"+"Render_{:04d}.jpg".format(i)
        #print(name)

        if(os.path.exists(mask_path)):
            mask_name = mask_path +"/"+"mask_{:04d}.jpg".format(i)
        else:
            os.mkdir(mask_path)
            mask_name = mask_path + "/"+"mask_{:04d}.jpg".format(i)
        mask = (mask>0)*255
        bg_idx  = random_bg_index = random.randint(0, 1500)
        bg_im = background_images[bg_idx]
        i = i +1
        cv2.imwrite(mask_name,mask)
        cv2.imwrite(rgb_name, 255*rgb)
        ren.clear()


