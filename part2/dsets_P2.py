
from collections import namedtuple
import functools
import glob
import numpy as np
from utils import xyz_tuple
from torch.utils.data import Dataset

import SimpleITK as sitk

candidate_info_tuple = namedtuple("candidate_info_tuple", "is_nodule_bool, diameter_mm, series_uid, center_xyz")

@functools.lru_cache(1)
def get_candidate_info_list(requires_on_disk=True):
    mhd_list = glob.glob("subset*/*.mhd")
    present_on_disk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open("annotations.csv", "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(annotation_center_xyz,
                                                            annotation_diameter_mm)
            
    candidate_info_list = []
    with open("candidates.csv", "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if (series_uid not in present_on_disk_set) and requires_on_disk:
                continue

            is_nodule_bool = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])

            candidate_diameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotation_center_xyz, annotation_diameter_mm = annotation_tup

                for i in range(3):
                    delta_mm = abs(candidate_center_xyz[i] - annotation_center_xyz[i])
                    if delta_mm > (annotation_diameter_mm/4):
                        break
                
                else: # Only runs if the above "for" loop executes without using "break"
                    candidate_diameter_mm = annotation_diameter_mm
                    break
            
            candidate_info_list.append(candidate_info_tuple(
                is_nodule_bool, candidate_diameter_mm, series_uid,
                center_xyz))
            
    candidate_info_list.sort(reverse=True)
    return candidate_info_list


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob("subset*/{}.mhd".format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        self.series_uid = series_uid
        self.hu_a = ct_a

        self.origin_xyz = xyz_tuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = xyz_tuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(*ct_mhd.GetDirection()).reshape(3,3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc

################################
def get_ct_augmented_candidate(augentation_dict,series_uid, center_xyz, width_irc, use_cache=True):

    if use_cache == True:
        ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)

    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    
    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_t = torch.eye(4)

    for i in range(3):
        if "flip" in augentation_dict:
            if random.random() > 0.5:
                transform_t[i,i] *= -1
            
        ## Shifting by random offset
        if "offset" in augmentation_dict:
            offset_float = augentation_dict["offset_float"]
            random_float = (random.random() * 2 - 1)
            transform_t[i,3] = offset_float * random_float
        
        ## Scaling (zoom in, zoom out)
        if "scale" in augmentation_dict:
            scale_float = augentation_dict["scale"]
            random_float = (random.random() * 2 - 1)
            transform_t[i,i] *= 1.0 + scale_float * random_float
        
        # NOTE: The values are always in [-1,1] range
    
    if "rotate" in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]])
        
        transform_t @= rotation_t
    
    affine_t = F.affine_grid(transform_t[:3].unsqueeze(0).to(torch.float32),ct_t.size(),align_corners=False)

    augmented_chunk = F.grid_sample(ct_t, affine_t, padding_mode='border', align_corners=False).to('cpu')

    if 'noise' in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc


class luna_dataset(Dataset):

    def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None, sortby_str='random', ratio_int=0, augmentation_dict=None, candidateInfo_list=None):

        



    pass




