
from collections import namedtuple
import numpy as np

irc_tuple = namedtuple("irc_tuple", ["index", "row", "col"])
xyz_tuple = namedtuple("xyz_tuple", ["x", "y", "z"])

def irc2xyz(coord_irc, origni_xyz, vx_size_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1] # flipping the matrix along "0" axis. np.flip(axis=0)
    origin_a = np.array(origni_xyz)
    vx_size_xyz_a = np.array(vx_size_xyz)

    coord_xyz = (direction_a @ (cri_a * vx_size_xyz_a)) + origin_a
    return xyz_tuple(*coord_xyz)

def xyz2irc(coord_xyz, origni_xyz, vx_size_xyz, direction_a):
    origin_a = np.array(origni_xyz)
    vx_size_a = np.array(vx_size_xyz)
    coord_a = np.array(coord_xyz)

    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vx_size_a
    cri_a = np.round(cri_a)
    
    return irc_tuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))



