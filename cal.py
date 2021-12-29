import copy
import numpy as np 
import sys
import pickle
import json
import cv2
import math
import os
import torch

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from pyquaternion import Quaternion
from functools import reduce
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes

from nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils import splits
from det3d.datasets.nuscenes.nusc_common import general_to_detection, _get_available_scenes, quaternion_yaw, _second_det_to_nusc_box, _lidar_nusc_box_to_global
from det3d.datasets.pipelines.loading import read_file
from det3d.core import box_np_ops
from det3d.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from configs.nusc.voxelnet.nusc_centerpoint_voxelnet_0075voxel_fix_bn_z import class_names
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.data_classes import Box

det2d_categories = ["car", "truck", "trailer", "bus", "construction_vehicle", "bicycle", "motorcycle", "pedestrian", "traffic_cone", "barrier"]
det2d_categories_dict = {k: i for i, k in enumerate(det2d_categories)}
det3d_categories_dict = {k: i for i, k in enumerate(class_names)}

def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points

def read_sweep(sweep, painted=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), painted=painted).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T

def load_points(info, painted, nsweeps=10):
    lidar_path = Path(info["lidar_path"])
    points = read_file(str(lidar_path), painted=painted)
    sweep_points_list = [points]
    sweep_times_list = [np.zeros((points.shape[0], 1))]

    for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
        sweep = info["sweeps"][i]
        points_sweep, times_sweep = read_sweep(sweep, painted=painted)
        sweep_points_list.append(points_sweep)
        sweep_times_list.append(times_sweep)

    points = np.concatenate(sweep_points_list, axis=0)
    times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

    return np.hstack([points, times])

def iou2d(a, b):
    lt = np.maximum(a[:, np.newaxis, :2], b[np.newaxis, :, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[np.newaxis, :, 2:])
    
    w = rb[..., 0] - lt[..., 0]
    h = rb[..., 1] - lt[..., 1]
    
    inter = w * h
    aarea = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    barea = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    overlaps = inter / np.clip(aarea[:, None] + barea[None, :] - inter, 1e-9, None)
    mask = (w <= 0) | (h <= 0)
    overlaps[mask] = 0
    return overlaps

def read_sensor(nusc, token):
    sensor_path, sensor_boxes, sensor_intrinsic = nusc.get_sample_data(token)
    sensor = nusc.get('sample_data', token)
    ego = nusc.get('ego_pose', sensor['ego_pose_token'])
    calib = nusc.get('calibrated_sensor', sensor['calibrated_sensor_token'])
    ego_from_sensor = transform_matrix(calib['translation'], Quaternion(calib['rotation']), inverse=False)
    global_from_ego = transform_matrix(ego['translation'], Quaternion(ego['rotation']), inverse=False)
    global_from_sensor = np.dot(global_from_ego, ego_from_sensor)
    sensor_from_ego = transform_matrix(calib['translation'], Quaternion(calib['rotation']), inverse=True)
    ego_from_global = transform_matrix(ego['translation'], Quaternion(ego['rotation']), inverse=True)
    sensor_from_global = np.dot(sensor_from_ego, ego_from_global)
    
    return sensor_path, sensor_boxes, sensor_intrinsic, sensor_from_global, global_from_sensor

def translate_box(nusc, box, sample, prev_sample):
    if sample is not None:
        lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego = nusc.get('ego_pose', lidar['ego_pose_token'])
        calib = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
    prev_lidar = nusc.get('sample_data', prev_sample['data']['LIDAR_TOP'])
    prev_ego = nusc.get('ego_pose', prev_lidar['ego_pose_token'])
    prev_calib = nusc.get('calibrated_sensor', prev_lidar['calibrated_sensor_token'])
    if sample is not None:
        box.rotate(Quaternion(calib['rotation']))
        box.translate(np.array(calib['translation']))
        box.rotate(Quaternion(ego['rotation']))
        box.translate(np.array(ego['translation']))
    box.translate(-np.array(prev_ego['translation']))
    box.rotate(Quaternion(prev_ego['rotation']).inverse)
    box.translate(-np.array(prev_calib['translation']))
    box.rotate(Quaternion(prev_calib['rotation']).inverse)
    return box

def match2d(cam_path, cam_boxes, cam_intrinsic, points, cam_from_g, g_from_cam):
    with open(os.path.join('data/nuScenes/detection2d/det_result', *(cam_path.split('/')[-3:])).replace('jpg', 'pkl'), 'rb') as f:
        det2d_result = pickle.load(f)

    cam_points = cam_from_g.dot(np.vstack((points, np.ones(points.shape[1]))))[:3, :]
    img_points = view_points(cam_points, cam_intrinsic, normalize=True)[:2, :]
    front = (cam_points[2, :] > 1) & (img_points[0, :] >= 0) & (img_points[0, :] < 1600) & (img_points[1, :] >= 0) & (img_points[1, :] < 900)
    img_points = img_points[:, front]
    cam_points = cam_points[:, front]
    #print(img_points.shape)
    img_boxes = []
    boxes_token = []
    boxes_names = []
    for box in cam_boxes:
        corners = box.corners()
        corners_img = view_points(corners, cam_intrinsic, normalize=True)[:2, :]
        corners_img[0, :] = np.clip(corners_img[0, :], 0, 1600-0.1)
        corners_img[1, :] = np.clip(corners_img[1, :], 0, 900-0.1)
        r,b = corners_img.max(1)
        l,t = corners_img.min(1)
        img_box = np.array([l,t,r,b])
        img_boxes.append(img_box)
        boxes_token.append(box.token)
        boxes_names.append(general_to_detection[box.name])
    img_boxes = np.array(img_boxes)
    boxes_token = np.array(boxes_token)
    boxes_names = np.array(boxes_names)

    det2d_boxes = []
    det2d_names = []
    max_overlaps = []
    token_assignment2d = []
    det2d_centers = []
    n = 0
    for i, det2d in enumerate(det2d_result):
        if len(det2d) == 0:
            continue
        det2d = det2d[det2d[:, -1] > 0.1]
        if len(det2d) > 0:
            gt_mask = (boxes_names == det2d_categories[i])
            det2d_name = np.array([det2d_categories[i] for j in range(len(det2d))])
            token = np.array(['' for j in range(len(det2d))])
            max_overlap = np.zeros((len(det2d),))
            if len(boxes_names) > 0 and gt_mask.sum() > 0:
                img_box = img_boxes[gt_mask]
                box_token = boxes_token[gt_mask]
                box_iou = iou2d(det2d[:, :4], img_box)
                max_overlap= np.max(box_iou, axis=1)
                gt_assignment = np.argmax(box_iou, axis=1)
                token = box_token[gt_assignment]
            det2d_boxes.append(det2d)
            det2d_names.append(det2d_name)
            max_overlaps.append(max_overlap)
            token_assignment2d.append(token)
            det2d_center = np.zeros((3, len(det2d)))
            for j, box in enumerate(det2d):
                #print(box)
                mask = (img_points[0, :] > box[0]) & (img_points[0, :] < box[2]) & (img_points[1, :] > box[1]) & (img_points[1, :] < box[3])
                if mask.sum() > 0:
                    cam_box_points = cam_points[:, mask]
                    det2d_center[:, j] = np.median(cam_box_points, axis=1)
            det2d_center = g_from_cam.dot(np.vstack((det2d_center, np.ones(len(det2d)))))[:3, :]
            det2d_centers.append(det2d_center)
            n += len(det2d)
    return det2d_boxes, det2d_names, max_overlaps, token_assignment2d, det2d_centers, int(n)

def generate_new_info(nusc, info, info_dict, det3d_result, det_point_root):
    sample = nusc.get('sample', info['token'])
    
    lidar_path, _, _, _, g_from_lidar = read_sensor(nusc, sample['data']['LIDAR_TOP'])
    camf_path, camf_boxes, camf_in, camf_from_g, g_from_camf = read_sensor(nusc, sample['data']['CAM_FRONT'])
    camfl_path, camfl_boxes, camfl_in, camfl_from_g, g_from_camfl = read_sensor(nusc, sample['data']['CAM_FRONT_LEFT'])
    camfr_path, camfr_boxes, camfr_in, camfr_from_g, g_from_camfr = read_sensor(nusc, sample['data']['CAM_FRONT_RIGHT'])
    camb_path, camb_boxes, camb_in, camb_from_g, g_from_camb = read_sensor(nusc, sample['data']['CAM_BACK'])
    cambl_path, cambl_boxes, cambl_in, cambl_from_g, g_from_cambl = read_sensor(nusc, sample['data']['CAM_BACK_LEFT'])
    cambr_path, cambr_boxes, cambr_in, cambr_from_g, g_from_cambr = read_sensor(nusc, sample['data']['CAM_BACK_RIGHT'])

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    if sample['prev'] == '':
        return None
    
    prev_sample = nusc.get('sample', sample['prev'])
    prev_info = info_dict[prev_sample['token']]
    _, _, _, lidar_from_g, _ = read_sensor(nusc, prev_sample['data']['LIDAR_TOP'])
    new_info = copy.deepcopy(prev_info)

    if not os.path.exists(det_point_root):
        os.mkdir(det_point_root)
    
    det3d = det3d_result['box3d_lidar'][:, :7, 0].numpy() # check which dim denotes rotation!!!
    offset = det3d_result['trajectories'][:, :, :2]
    offset = offset.sum(dim=1).numpy()
    labels = det3d_result['label_preds'].numpy()
    scores = det3d_result['scores'].numpy()
    scores_rect = det3d_result['score_rect'].numpy()
    names = np.array([class_names[int(label)] for label in labels])
    num_obj  = det3d.shape[0]
    print(num_obj)

    #all_points = load_points(info, painted=False)
    #point_indices = box_np_ops.points_in_rbbox(all_points, det3d)
    enlarge_det3d = deepcopy(det3d)
    enlarge_det3d[:, 3:6] *= 1.25
    #enlarge_point_indices = box_np_ops.points_in_rbbox(all_points, enlarge_det3d)
    det_point_path = []
    det_point_box = []
    cur_det_path = os.path.join(det_point_root, info['token'])
    if not os.path.exists(cur_det_path):
        os.mkdir(cur_det_path)

    for i in range(num_obj):
        '''
        obj_points = all_points[point_indices[:, i]]
        if len(obj_points) > 0:
            onehot = np.eye(11)[int(labels[i]+1)]
            onehot = np.tile(onehot, (len(obj_points), 1))
            obj_points = np.concatenate([obj_points, onehot], axis=1)
        extra_points = all_points[((~point_indices[:, i]) & enlarge_point_indices[:, i])]
        if len(extra_points) > 0:
            onehot = np.zeros((len(extra_points), 11))
            extra_points = np.concatenate([extra_points, onehot], axis=1)
            if len(obj_points) > 0:
                obj_points = np.concatenate([obj_points, extra_points], axis=0)
            else:
                obj_points = extra_points
        if len(obj_points) > 0:
            obj_points = obj_points.T
            obj_points[:3, :] = g_from_lidar.dot(np.vstack((obj_points[:3, :], np.ones(obj_points.shape[1]))))[:3, :]
            obj_points = obj_points.T.astype(np.float32)
        else:
            obj_points = np.zeros((1,16), dtype=np.float32)
        '''
        det_point_path.append(os.path.join(cur_det_path, '%s_%d.bin'%(class_names[labels[i]], i)))
        #obj_points.tofile(det_point_path[-1])
        box = Box(
            det3d[i, :3],
            det3d[i, 3:6],
            Quaternion(axis=[0,0,1], radians=(-det3d[i, 6]-np.pi/2)),
            label=labels[i],
            score=scores[i],
            velocity=(*offset[i, :], 0.0)
        )
        box = translate_box(nusc, box, sample, prev_sample)
        det_point_box.append(box)
    locs = np.array([b.center for b in det_point_box]).reshape(-1, 3)
    dims = np.array([b.wlh for b in det_point_box]).reshape(-1, 3)
    velocity = np.array([b.velocity for b in det_point_box]).reshape(-1, 3)
    rots = np.array([quaternion_yaw(b.orientation) for b in det_point_box]).reshape(-1, 1)
    det_box3d = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
    
    gt_names = info['gt_names']
    gt_boxes = info['gt_boxes'][:, [0,1,2,3,4,5,-1]]
    max_overlaps = np.zeros((len(det3d),), dtype=np.float32)
    enlarge_max_overlaps = np.zeros((len(det3d),), dtype=np.float32)
    gt_assignment3d = np.zeros((len(det3d),), dtype=np.int32)
    if len(gt_names) > 0:
        for c in class_names:
            det_mask = (names == c)
            gt_mask = (gt_names == c)
            if det_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_det = torch.from_numpy(det3d[det_mask]).float()
                cur_gt = torch.from_numpy(gt_boxes[gt_mask]).float()
                non_zero_gt = gt_mask.nonzero()[0]

                iou3d = boxes_iou3d_gpu(cur_det.cuda(), cur_gt.cuda())
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d.cpu(), dim=1)
                max_overlaps[det_mask] = cur_max_overlaps.numpy()
                gt_assignment3d[det_mask] = non_zero_gt[cur_gt_assignment.numpy()]

                cur_det[:, 3:6] *= 1.25
                enlarge_overlaps = boxes_iou3d_gpu(cur_det.cuda(), cur_gt.cuda(), overlap=True).cpu()
                enlarge_overlaps = enlarge_overlaps[torch.arange(cur_det.shape[0]), cur_gt_assignment].numpy()
                enlarge_overlaps = enlarge_overlaps / (cur_det[:, 3] * cur_det[:, 4] * cur_det[:, 5]).numpy() * (1.25**3)
                enlarge_max_overlaps[det_mask] = enlarge_overlaps
    if len(info['gt_boxes_token']) == 0:
        token_assignment3d = np.array(['' for i in range(det_box3d.shape[0])])
    else: 
        token_assignment3d = info['gt_boxes_token'][gt_assignment3d]
    info_3d = dict(
        det_box3d=det_box3d.astype(np.float32), det_velocity=velocity.astype(np.float32), det_names = names, det_score=scores.astype(np.float32), det_score_rect = scores_rect.astype(np.float32),
        det_iou=max_overlaps.astype(np.float32), det_iou_enlarge=enlarge_max_overlaps.astype(np.float32), det_point_path=det_point_path
    )
    new_info.update(info_3d)

    global_points = g_from_lidar.dot(np.vstack((points[:, :3].T, np.ones(points.shape[0]))))[:3, :]

    camf2d_boxes, camf2d_names, camf_max_overlaps, camf_token, camf2d_centers, fn = match2d(
        camf_path, camf_boxes, camf_in, global_points, camf_from_g, g_from_camf
    )
    camfl2d_boxes, camfl2d_names, camfl_max_overlaps, camfl_token, camfl2d_centers, fln = match2d(
        camfl_path, camfl_boxes, camfl_in, global_points, camfl_from_g, g_from_camfl
    )
    camfr2d_boxes, camfr2d_names, camfr_max_overlaps, camfr_token, camfr2d_centers, frn = match2d(
        camfr_path, camfr_boxes, camfr_in, global_points, camfr_from_g, g_from_camfr
    )
    camb2d_boxes, camb2d_names, camb_max_overlaps, camb_token, camb2d_centers, bn = match2d(
        camb_path, camb_boxes, camb_in, global_points, camb_from_g, g_from_camb
    )
    cambl2d_boxes, cambl2d_names, cambl_max_overlaps, cambl_token, cambl2d_centers, bln = match2d(
        cambl_path, cambl_boxes, cambl_in, global_points, cambl_from_g, g_from_cambl
    )
    cambr2d_boxes, cambr2d_names, cambr_max_overlaps, cambr_token, cambr2d_centers, brn = match2d(
        cambr_path, cambr_boxes, cambr_in, global_points, cambr_from_g, g_from_cambr
    )
    det2d_boxes = camf2d_boxes + camfl2d_boxes + camfr2d_boxes + camb2d_boxes + cambl2d_boxes + cambr2d_boxes
    det2d_names = camf2d_names + camfl2d_names + camfr2d_names + camb2d_names + cambl2d_names + cambr2d_names
    det2d_max_overlaps = camf_max_overlaps + camfl_max_overlaps + camfr_max_overlaps + camb_max_overlaps + cambl_max_overlaps + cambr_max_overlaps
    det2d_token = camf_token + camfl_token + camfr_token + camb_token + cambl_token + cambr_token
    det2d_centers = camf2d_centers + camfl2d_centers + camfr2d_centers + camb2d_centers + cambl2d_centers + cambr2d_centers
    print(len(det2d_boxes))
    if len(det2d_boxes) == 0:
        img_idx = np.array([0])
        det2d_boxes = np.zeros((1, 5))
        det2d_names = np.array(['car'])
        det2d_max_overlaps = np.array([0.])
        det2d_token = np.array([''])
        det2d_centers = np.zeros((1, 3))
    else:
        img_idx = np.concatenate([np.zeros(fn), np.ones(fln), np.ones(frn)*2, np.ones(bn)*3, np.ones(bln)*4, np.ones(brn)*5])
        det2d_boxes = np.concatenate(det2d_boxes, axis=0)
        det2d_names = np.concatenate(det2d_names, axis=0)
        det2d_max_overlaps = np.concatenate(det2d_max_overlaps, axis=0)
        det2d_token = np.concatenate(det2d_token, axis=0)
        det2d_centers = np.concatenate(det2d_centers, axis=1)
        det2d_centers = lidar_from_g.dot(np.vstack((det2d_centers, np.ones(det2d_centers.shape[1]))))[:3, :].T

    info_2d = dict(
        det_box_img=det2d_boxes[:, :4].astype(np.float32), det_score_img=det2d_boxes[:, -1].astype(np.float32), det_name_img=det2d_names,
        det_iou2d=det2d_max_overlaps.astype(np.float32), det2d_centers=det2d_centers.astype(np.float32), img_idx=img_idx,
        det_img_path=[camf_path, camfl_path, camfr_path, camb_path, cambl_path, cambr_path]
    )
    new_info.update(info_2d)

    box_dict = {}
    for k in info['gt_boxes_token']:
        if k != '':
            box_dict[k] = dict(idx_3d=-1, iou_3d=0.4, idx_2d=[], idx_prev=-1)
    for k in det2d_token:
        if k not in box_dict.keys() and (k != ''):
            box_dict[k] = dict(idx_3d=-1, iou_3d=0.4, idx_2d=[], idx_prev=-1)
    for i, k in enumerate(prev_info['gt_boxes_token']):
        anno = nusc.get('sample_annotation', k)
        if anno['next'] != '':
            if anno['next'] not in box_dict.keys():
                box_dict[anno['next']] = dict(idx_3d=-1, iou_3d=0.4, idx_2d=[], idx_prev=-1)
            box_dict[anno['next']]['idx_prev'] = i

    prev_assignment3d = np.ones(len(token_assignment3d), dtype=np.int32) * (-1)
    for i, t in enumerate(token_assignment3d):
        if t == '':
            continue
        if enlarge_max_overlaps[i] > 0.5:
            prev_assignment3d[i] = box_dict[t]['idx_prev']
        if enlarge_max_overlaps[i] > box_dict[t]['iou_3d']:
            box_dict[t]['idx_3d'] = i
            box_dict[t]['iou_3d'] = enlarge_max_overlaps[i]
    
    prev_assignment2d = np.ones(len(det2d_token), dtype=np.int32) * (-1)
    for i, t in enumerate(det2d_token):
        if det2d_max_overlaps[i] > 0.4:
            prev_assignment2d[i] = box_dict[t]['idx_prev']
            box_dict[t]['idx_2d'].append(i)
    
    match_tuple = []
    for v in box_dict.values():
        if (v['idx_3d'] >= 0) and (len(v['idx_2d']) > 0):
            match_tuple.append((v['idx_3d'], v['idx_2d']))
    match_array = np.ones((len(match_tuple), 3), dtype=np.int32) * (-1)
    for i, (x1, x2) in enumerate(match_tuple):
        match_array[i, 0] = int(x1)
        match_array[i, 1] = int(x2[0])
        if len(x2) > 1:
            match_array[i, 2] = int(x2[1])

    gt_dict = dict(
        gt_assignment3d=prev_assignment3d, gt_assignment2d=prev_assignment2d, match_array=match_array
    )
    new_info.update(gt_dict)
    return new_info

def generate_new_info_v2(nusc, info, info_dict, det3d_result, det_point_root):
    sample = nusc.get('sample', info['token'])
    
    lidar_path, _, _, _, g_from_lidar = read_sensor(nusc, sample['data']['LIDAR_TOP'])
    camf_path, camf_boxes, camf_in, camf_from_g, g_from_camf = read_sensor(nusc, sample['data']['CAM_FRONT'])
    camfl_path, camfl_boxes, camfl_in, camfl_from_g, g_from_camfl = read_sensor(nusc, sample['data']['CAM_FRONT_LEFT'])
    camfr_path, camfr_boxes, camfr_in, camfr_from_g, g_from_camfr = read_sensor(nusc, sample['data']['CAM_FRONT_RIGHT'])
    camb_path, camb_boxes, camb_in, camb_from_g, g_from_camb = read_sensor(nusc, sample['data']['CAM_BACK'])
    cambl_path, cambl_boxes, cambl_in, cambl_from_g, g_from_cambl = read_sensor(nusc, sample['data']['CAM_BACK_LEFT'])
    cambr_path, cambr_boxes, cambr_in, cambr_from_g, g_from_cambr = read_sensor(nusc, sample['data']['CAM_BACK_RIGHT'])

    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    if sample['prev'] == '':
        return None
    
    next_sample = nusc.get('sample', sample['prev'])
    next_info = info_dict[next_sample['token']]
    _, _, _, lidar_from_g, _ = read_sensor(nusc, next_sample['data']['LIDAR_TOP'])
    new_info = copy.deepcopy(next_info)

    if not os.path.exists(det_point_root):
        os.mkdir(det_point_root)
    
    det3d = det3d_result['box3d_lidar'][:, :7, 0].numpy() # check which dim denotes rotation!!!
    offset = det3d_result['trajectories'][:, :, :2]
    offset = offset.sum(dim=1).numpy()
    labels = det3d_result['label_preds'].numpy()
    scores = det3d_result['scores'].numpy()
    scores_rect = det3d_result['score_rect'].numpy()
    names = np.array([class_names[int(label)] for label in labels])
    num_obj  = det3d.shape[0]
    #print(num_obj)

    all_points = load_points(info, painted=False)
    point_indices = box_np_ops.points_in_rbbox(all_points, det3d)
    enlarge_det3d = deepcopy(det3d)
    enlarge_det3d[:, 3:6] *= 1.25
    enlarge_point_indices = box_np_ops.points_in_rbbox(all_points, enlarge_det3d)
    det_point_path = []
    det_point_box = []
    cur_det_path = os.path.join(det_point_root, info['token'])
    if not os.path.exists(cur_det_path):
        os.mkdir(cur_det_path)

    for i in range(num_obj):
        det_point_path.append(os.path.join(cur_det_path, '%s_%d.bin'%(class_names[labels[i]], i)))
        if not os.path.exists(det_point_path[-1]):
            print(info['token'], i)
            obj_points = all_points[point_indices[:, i]]
            if len(obj_points) > 0:
                onehot = np.eye(11)[int(labels[i]+1)]
                onehot = np.tile(onehot, (len(obj_points), 1))
                obj_points = np.concatenate([obj_points, onehot], axis=1)
            extra_points = all_points[((~point_indices[:, i]) & enlarge_point_indices[:, i])]
            if len(extra_points) > 0:
                onehot = np.zeros((len(extra_points), 11))
                extra_points = np.concatenate([extra_points, onehot], axis=1)
                if len(obj_points) > 0:
                    obj_points = np.concatenate([obj_points, extra_points], axis=0)
                else:
                    obj_points = extra_points
            if len(obj_points) > 0:
                obj_points = obj_points.T
                obj_points[:3, :] = g_from_lidar.dot(np.vstack((obj_points[:3, :], np.ones(obj_points.shape[1]))))[:3, :]
                obj_points = obj_points.T.astype(np.float32)
            else:
                obj_points = np.zeros((1,16), dtype=np.float32)
            obj_points.tofile(det_point_path[-1])
        box = Box(
            det3d[i, :3],
            det3d[i, 3:6],
            Quaternion(axis=[0,0,1], radians=(-det3d[i, 6]-np.pi/2)),
            label=labels[i],
            score=scores[i],
            velocity=(*offset[i, :], 0.0)
        )
        box = translate_box(nusc, box, sample, next_sample)
        det_point_box.append(box)
    locs = np.array([b.center for b in det_point_box]).reshape(-1, 3)
    dims = np.array([b.wlh for b in det_point_box]).reshape(-1, 3)
    #velocity = np.array([b.velocity for b in det_point_box]).reshape(-1, 3)
    rots = np.array([quaternion_yaw(b.orientation) for b in det_point_box]).reshape(-1, 1)
    det_box3d = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
    
    gt_names = info['gt_names']
    gt_boxes = info['gt_boxes'][:, [0,1,2,3,4,5,-1]]
    full_boxes = nusc.get_boxes(sample['data']['LIDAR_TOP'])
    gt_boxes_list = []
    for box in full_boxes:
        box.velocity = nusc.box_velocity(box.token)
        box = translate_box(nusc, box, None, next_sample)
        gt_boxes_list.append(box)
    locs = np.array([b.center for b in gt_boxes_list]).reshape(-1, 3)
    dims = np.array([b.wlh for b in gt_boxes_list]).reshape(-1, 3)
    velocity = np.array([b.velocity for b in gt_boxes_list]).reshape(-1, 3)
    rots = np.array([quaternion_yaw(b.orientation) for b in gt_boxes_list]).reshape(-1, 1)
    gt_boxes_next = np.concatenate([locs, dims, velocity[:, :2], -rots - np.pi / 2], axis=1)
    full_gt_tokens = np.array([box.token for box in gt_boxes_list])

    max_overlaps = np.zeros((len(det3d),), dtype=np.float32)
    enlarge_max_overlaps = np.zeros((len(det3d),), dtype=np.float32)
    gt_assignment3d = np.zeros((len(det3d),), dtype=np.int32)
    if len(gt_names) > 0:
        for c in class_names:
            det_mask = (names == c)
            gt_mask = (gt_names == c)
            if det_mask.sum() > 0 and gt_mask.sum() > 0:
                cur_det = torch.from_numpy(det3d[det_mask]).float()
                cur_gt = torch.from_numpy(gt_boxes[gt_mask]).float()
                non_zero_gt = gt_mask.nonzero()[0]

                iou3d = boxes_iou3d_gpu(cur_det.cuda(), cur_gt.cuda())
                cur_max_overlaps, cur_gt_assignment = torch.max(iou3d.cpu(), dim=1)
                max_overlaps[det_mask] = cur_max_overlaps.numpy()
                gt_assignment3d[det_mask] = non_zero_gt[cur_gt_assignment.numpy()]

                cur_det[:, 3:6] *= 1.25
                enlarge_overlaps = boxes_iou3d_gpu(cur_det.cuda(), cur_gt.cuda(), overlap=True).cpu()
                enlarge_overlaps = enlarge_overlaps[torch.arange(cur_det.shape[0]), cur_gt_assignment].numpy()
                enlarge_overlaps = enlarge_overlaps / (cur_det[:, 3] * cur_det[:, 4] * cur_det[:, 5]).numpy() * (1.25**3)
                enlarge_max_overlaps[det_mask] = enlarge_overlaps
    if len(info['gt_boxes_token']) == 0:
        token_assignment3d = np.array(['' for i in range(det_box3d.shape[0])])
    else: 
        token_assignment3d = info['gt_boxes_token'][gt_assignment3d]

    global_points = g_from_lidar.dot(np.vstack((points[:, :3].T, np.ones(points.shape[0]))))[:3, :]

    camf2d_boxes, camf2d_names, camf_max_overlaps, camf_token, camf2d_centers, fn = match2d(
        camf_path, camf_boxes, camf_in, global_points, camf_from_g, g_from_camf
    )
    camfl2d_boxes, camfl2d_names, camfl_max_overlaps, camfl_token, camfl2d_centers, fln = match2d(
        camfl_path, camfl_boxes, camfl_in, global_points, camfl_from_g, g_from_camfl
    )
    camfr2d_boxes, camfr2d_names, camfr_max_overlaps, camfr_token, camfr2d_centers, frn = match2d(
        camfr_path, camfr_boxes, camfr_in, global_points, camfr_from_g, g_from_camfr
    )
    camb2d_boxes, camb2d_names, camb_max_overlaps, camb_token, camb2d_centers, bn = match2d(
        camb_path, camb_boxes, camb_in, global_points, camb_from_g, g_from_camb
    )
    cambl2d_boxes, cambl2d_names, cambl_max_overlaps, cambl_token, cambl2d_centers, bln = match2d(
        cambl_path, cambl_boxes, cambl_in, global_points, cambl_from_g, g_from_cambl
    )
    cambr2d_boxes, cambr2d_names, cambr_max_overlaps, cambr_token, cambr2d_centers, brn = match2d(
        cambr_path, cambr_boxes, cambr_in, global_points, cambr_from_g, g_from_cambr
    )
    det2d_boxes = camf2d_boxes + camfl2d_boxes + camfr2d_boxes + camb2d_boxes + cambl2d_boxes + cambr2d_boxes
    det2d_names = camf2d_names + camfl2d_names + camfr2d_names + camb2d_names + cambl2d_names + cambr2d_names
    det2d_max_overlaps = camf_max_overlaps + camfl_max_overlaps + camfr_max_overlaps + camb_max_overlaps + cambl_max_overlaps + cambr_max_overlaps
    det2d_token = camf_token + camfl_token + camfr_token + camb_token + cambl_token + cambr_token
    det2d_centers = camf2d_centers + camfl2d_centers + camfr2d_centers + camb2d_centers + cambl2d_centers + cambr2d_centers
    #print(len(det2d_boxes))
    if len(det2d_boxes) == 0:
        img_idx = np.array([0])
        det2d_boxes = np.zeros((1, 5))
        det2d_names = np.array(['car'])
        det2d_max_overlaps = np.array([0.])
        det2d_token = np.array([''])
        det2d_centers = np.zeros((1, 3))
    else:
        img_idx = np.concatenate([np.zeros(fn), np.ones(fln), np.ones(frn)*2, np.ones(bn)*3, np.ones(bln)*4, np.ones(brn)*5])
        det2d_boxes = np.concatenate(det2d_boxes, axis=0)
        det2d_names = np.concatenate(det2d_names, axis=0)
        det2d_max_overlaps = np.concatenate(det2d_max_overlaps, axis=0)
        det2d_token = np.concatenate(det2d_token, axis=0)
        det2d_centers = np.concatenate(det2d_centers, axis=1)
        det2d_centers = lidar_from_g.dot(np.vstack((det2d_centers, np.ones(det2d_centers.shape[1]))))[:3, :].T

    

    box_dict = {}
    for i, k in enumerate(full_gt_tokens):
        if k != '':
            box_dict[k] = dict(idx_3d=-1, iou_3d=0.4, idx_2d=[], idx_next=-1, idx=i)
    for i, k in enumerate(next_info['gt_boxes_token']):
        anno = nusc.get('sample_annotation', k)
        if anno['next'] in box_dict.keys():
            box_dict[anno['next']]['idx_next'] = i

    prev_assignment3d = np.ones((len(token_assignment3d),2), dtype=np.int32) * (-1)
    for i, t in enumerate(token_assignment3d):
        if t == '':
            continue
        prev_assignment3d[i, 0] = box_dict[t]['idx']
        prev_assignment3d[i, 1] = box_dict[t]['idx_next']
        if enlarge_max_overlaps[i] > box_dict[t]['iou_3d']:
            box_dict[t]['idx_3d'] = i
            box_dict[t]['iou_3d'] = enlarge_max_overlaps[i]
    
    prev_assignment2d = np.ones((len(det2d_token),2), dtype=np.int32) * (-1)
    for i, t in enumerate(det2d_token):
        if t == '':
            continue
        prev_assignment2d[i, 0] = box_dict[t]['idx']
        prev_assignment2d[i, 1] = box_dict[t]['idx_next']
        if det2d_max_overlaps[i] > 0.5:
            box_dict[t]['idx_2d'].append(i)
    
    match_tuple = []
    for v in box_dict.values():
        if (v['idx_3d'] >= 0) and (len(v['idx_2d']) > 0):
            match_tuple.append((v['idx_3d'], v['idx_2d']))
    match_array = np.ones((len(match_tuple), 3), dtype=np.int32) * (-1)
    for i, (x1, x2) in enumerate(match_tuple):
        match_array[i, 0] = int(x1)
        match_array[i, 1] = int(x2[0])
        if len(x2) > 1:
            match_array[i, 2] = int(x2[1])

    info_3d = dict(
        det_box3d=det_box3d.astype(np.float32), det_names=names, det_score=scores.astype(np.float32), det_score_rect = scores_rect.astype(np.float32),
        det_iou=max_overlaps.astype(np.float32), det_iou_enlarge=enlarge_max_overlaps.astype(np.float32), det_point_path=det_point_path, det_velocity=gt_boxes_next
    )
    new_info.update(info_3d)
    info_2d = dict(
        det_box_img=det2d_boxes[:, :4].astype(np.float32), det_score_img=det2d_boxes[:, -1].astype(np.float32), det_name_img=det2d_names,
        det_iou2d=det2d_max_overlaps.astype(np.float32), det2d_centers=det2d_centers.astype(np.float32), img_idx=img_idx,
        det_img_path=[camf_path, camfl_path, camfr_path, camb_path, cambl_path, cambr_path]
    )
    new_info.update(info_2d)
    gt_dict = dict(
        gt_assignment3d=prev_assignment3d, gt_assignment2d=prev_assignment2d, match_array=match_array
    )
    new_info.update(gt_dict)
    return new_info

def get_frustum(nusc, infos):
    err = []
    gt = []
    det = []
    for info in infos:
        token = info['token']
        gt_boxes_token = info['gt_boxes_token']
        gt_assignment2d = info['gt_assignment2d']
        det_iou2d = info['det_iou2d']
        det2d_centers = info['det2d_centers']
        mask = (gt_assignment2d >= 0) & (det_iou2d > 0.5)
        gt2d_tokens = gt_boxes_token[gt_assignment2d[mask]]
        det2d_centers = det2d_centers[mask]

        sample = nusc.get('sample', token)
        lidar = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego = nusc.get('ego_pose', lidar['ego_pose_token'])
        calib = nusc.get('calibrated_sensor', lidar['calibrated_sensor_token'])
        for gt2d_token, det2d_center in zip(gt2d_tokens, det2d_centers):
            anno = nusc.get('sample_annotation', gt2d_token)
            box = nusc.get_box(anno['next'])
            box.translate(-np.array(ego['translation']))
            box.rotate(Quaternion(ego['rotation']).inverse)
            box.translate(-np.array(calib['translation']))
            box.rotate(Quaternion(calib['rotation']).inverse)
            gt2d_center = np.array(box.center)
            err.append(gt2d_center[:2]-det2d_center[:2])
            det.append(det2d_center[:2])
            gt.append(gt2d_center[:2])
    np.save('err.npy', np.array(err))
    np.save('det.npy', np.array(det))
    np.save('gt.npy', np.array(gt))


if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuScenes', verbose=True)
  
    with open('data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl', 'rb') as f:
        infos = pickle.load(f)

    info_dict = {info['token']: info for info in infos}
    with open('work_dirs/nusc_tracking_rect_center_sum/prediction.pkl', 'rb') as f:
        det3d_results = pickle.load(f)
    
    new_infos = []
    for i in tqdm(range(len(infos))):
        new_info = generate_new_info_v2(nusc, infos[i], info_dict, det3d_results[infos[i]['token']], 'data/nuScenes/obj_points')
        if new_info is not None:
            new_infos.append(new_info)
    with open('data/nuScenes/infos_train_track_v3.pkl', 'wb') as f:
        pickle.dump(new_infos, f)

    '''
    with open('data/nuScenes/infos_train_track.pkl', 'rb') as f:
        infos = pickle.load(f)
    get_frustum(nusc, infos)
    '''
