import math
import numpy as np
import cv2
from tqdm import tqdm
from scipy.linalg import expm3, norm

from renderer import Renderer

def Get_Rx(ax):
    ax = ax * math.pi / 180.0
    return np.asarray([[1,0,0],[0,math.cos(ax),-math.sin(ax)],[0,math.sin(ax),math.cos(ax)]])

def Get_Ry(ay):
    ay = ay * math.pi / 180.0
    return np.asarray([[math.cos(ay),0,math.sin(ay)],[0,1,0],[-math.sin(ay),0,math.cos(ay)]])

def Get_Rz(az):
    az = az * math.pi / 180.0
    return np.asarray([[math.cos(az),-math.sin(az),0],[math.sin(az),math.cos(az),0],[0,0,1]])


def create_pose_using_Euler_angle(vertex):

    angle_x = vertex[0]
    angle_y = vertex[1]
    angle_z = vertex[2]
    rotation = Get_Rx(angle_x)*Get_Ry(angle_y)*Get_Rz(angle_z)


def compute_rotation_from_vertex(vertex):
    """Compute rotation matrix from viewpoint vertex """
    up = [0, 0, 1]
    if vertex[0] == 0 and vertex[1] == 0 and vertex[2] != 0:
        up = [-1, 0, 0]
    rot = np.zeros((3, 3))
    rot[:, 2] = -vertex / norm(vertex)  # View direction towards origin
    rot[:, 0] = np.cross(rot[:, 2], up)
    rot[:, 0] /= norm(rot[:, 0])
    rot[:, 1] = np.cross(rot[:, 0], -rot[:, 2])
    return rot.T


def create_pose(vertex, scale=0, angle_deg=0):
    """Compute rotation matrix from viewpoint vertex and inplane rotation """
    rot = compute_rotation_from_vertex(vertex)
    transform = np.eye(4)
    rodriguez = np.asarray([0, 0, 1]) * (angle_deg * math.pi / 180.0)
    angle_axis = expm3(np.cross(np.eye(3), rodriguez))
    transform[0:3, 0:3] = np.matmul(angle_axis, rot)
    transform[0:3, 3] = [0, 0, scale]
    return transform


def precompute_projections(views, inplanes, cam, model3D):
    """Precomputes the projection information needed for 6D pose construction

    # Arguments
        views: List of 3D viewpoint positions
        inplanes: List of inplane angles in degrees
        cam: Intrinsics to use for translation estimation
        model3D: Model3D instance

    # Returns
        data: a 3D list with precomputed entities with shape
            (views, inplanes, (4x4 pose matrix, 3) )
    """
    w, h = 640, 480
    ren = Renderer((w, h), cam)
    data = []
    if model3D.vertices is None:
        return data

    for v in tqdm(range(len(views))):
        data.append([])
        for i in inplanes:
            pose = create_pose(views[v], angle_deg=i)
            pose[:3, 3] = [0, 0, 0.5]  # zr = 0.5

            # Render object and extract tight 2D bbox and projected 2D centroid
            ren.clear()
            ren.draw_model(model3D, pose)
            box = np.argwhere(ren.finish()[1])  # Deduct bbox from depth rendering
            box = [box.min(0)[1], box.min(0)[0], box.max(0)[1] + 1, box.max(0)[0] + 1]
            centroid = np.matmul(pose[:3, :3], model3D.centroid) + pose[:3, 3]
            centroid_x = cam[0, 2] + centroid[0] * cam[0, 0] / centroid[2]
            centroid_y = cam[1, 2] + centroid[1] * cam[1, 1] / centroid[2]

            # Compute 2D centroid position in normalized, box-local reference frame
            box_w, box_h = (box[2] - box[0]), (box[3] - box[1])
            norm_centroid_x = (centroid_x - box[0]) / box_w
            norm_centroid_y = (centroid_y - box[1]) / box_h

            # Compute normalized diagonal box length
            lr = np.sqrt((box_w / w) ** 2 + (box_h / h) ** 2)
            data[-1].append((pose, [norm_centroid_x, norm_centroid_y, lr]))
    return data

