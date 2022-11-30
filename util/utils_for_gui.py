import open3d as o3d
import plyfile
import torch
import time
import numpy as np
import random
import os
from util.config import cfg
cfg.task = 'test'
import util.eval as eval
from operator import itemgetter
from PIL import Image


COLOR20 = np.array(
        [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
        [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
        [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
        [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

COLOR40 = np.array(
        [[88,170,108], [174,105,226], [78,194,83], [198,62,165], [133,188,52], [97,101,219], [190,177,52], [139,65,168], [75,202,137], [225,66,129],
        [68,135,42], [226,116,210], [146,186,98], [68,105,201], [219,148,53], [85,142,235], [212,85,42], [78,176,223], [221,63,77], [68,195,195],
        [175,58,119], [81,175,144], [184,70,74], [40,116,79], [184,134,219], [130,137,46], [110,89,164], [92,135,74], [220,140,190], [94,103,39],
        [144,154,219], [160,86,40], [67,107,165], [194,170,104], [162,95,150], [143,110,44], [146,72,105], [225,142,106], [162,83,86], [227,124,143]])

SEMANTIC_IDXS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
SEMANTIC_NAMES = np.array(['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter',
                        'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'])
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'floor': [143, 223, 142],
    'wall': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160]
}
SEMANTIC_IDX2NAME = {1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair', 6: 'sofa', 7: 'table', 8: 'door', 9: 'window', 10: 'bookshelf', 11: 'picture',
                12: 'counter', 14: 'desk', 16: 'curtain', 24: 'refridgerator', 28: 'shower curtain', 33: 'toilet',  34: 'sink', 36: 'bathtub', 39: 'otherfurniture'}


def ply2pth(fn):
    f = plyfile.PlyData().read(fn)
    # x,y,z,r,g,b,nx,ny,nz,curvature
    points = np.array([list(x) for x in f.elements[0]], dtype=np.float32)
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
    # --- from raw ply file to torch format
    target = os.path.join('/tmp', 'scans', 'test')
    os.makedirs(target, exist_ok=True)
    target_fn = os.path.join(target, os.path.basename(fn).replace('.ply', '_inst_nostuff.pth'))
    torch.save((coords, colors), target_fn)


def init():
    global result_dir
    result_dir = os.path.join(cfg.data_root, 'result')
    torch.cuda.empty_cache()
    global semantic_label_idx
    semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)


def test(model, model_fn, epoch):
    from data.scannetv2_inst import Dataset
    dataset = Dataset(test=True)
    dataset.testLoader()
    dataloader = dataset.test_data_loader
    with torch.no_grad():
        model = model.eval()
        start = time.time()
        matches = {}
        for i, batch in enumerate(dataloader):
            N = batch['feats'].shape[0]
            test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1][:-17]
            start1 = time.time()
            preds = model_fn(batch, model, epoch)
            end1 = time.time() - start1
            ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
            semantic_scores = preds['semantic']  # (N, nClass=20) float32, cuda
            semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda
            if (epoch > cfg.prepare_epochs):
                scores = preds['score']   # (nProposal, 1) float, cuda
                scores_pred = torch.sigmoid(scores.view(-1))
                proposals_idx, proposals_offset = preds['proposals']
                proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores_pred.device) # (nProposal, N), int, cuda
                proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
                semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device)[semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]]
                ##### score threshold
                score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
                scores_pred = scores_pred[score_mask]
                proposals_pred = proposals_pred[score_mask]
                semantic_id = semantic_id[score_mask]
                ##### npoint threshold
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)
                scores_pred = scores_pred[npoint_mask]
                proposals_pred = proposals_pred[npoint_mask]
                semantic_id = semantic_id[npoint_mask]
                ##### nms
                if semantic_id.shape[0] == 0:
                    pick_idxs = np.empty(0)
                else:
                    proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
                    intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                    proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                    cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                    pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(), cfg.TEST_NMS_THRESH)  # int, (nCluster, N)
                clusters = proposals_pred[pick_idxs]
                cluster_scores = scores_pred[pick_idxs]
                cluster_semantic_id = semantic_id[pick_idxs]
                nclusters = clusters.shape[0]
                ##### prepare for evaluation
                if cfg.eval:
                    pred_info = {}
                    pred_info['conf'] = cluster_scores.cpu().numpy()
                    pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                    pred_info['mask'] = clusters.cpu().numpy()
                    gt_file = os.path.join(cfg.data_root, cfg.dataset, cfg.split + '_gt', test_scene_name + '.txt')
                    gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)
                    matches[test_scene_name] = {}
                    matches[test_scene_name]['gt'] = gt2pred
                    matches[test_scene_name]['pred'] = pred2gt
            ##### save files
            start3 = time.time()
            if cfg.save_semantic:
                os.makedirs(os.path.join(result_dir, 'semantic_npy'), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                np.save(os.path.join(result_dir, 'semantic_npy', test_scene_name + '.npy'), semantic_np)
            end3 = time.time() - start3
            end = time.time() - start
            start = time.time()
    return cluster_semantic_id


def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def get_raw_coords_color(input_file):
    f = plyfile.PlyData().read(input_file)
    points = np.array([list(x) for x in f.elements[0]])
    xyz = np.ascontiguousarray(points[:, :3])
    rgb = np.ascontiguousarray(points[:, 3:6])
    return xyz, rgb


def get_semantic(semantic_file):
    rgb_out = None
    label_pred = np.load(semantic_file).astype(np.int)  # 0~19
    label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR)).astype(np.int)
    if np.shape(rgb_out) == ():
        rgb_out = label_pred_rgb
    else:
        rgb_out = np.concatenate((rgb_out,label_pred_rgb),axis=0)
    return rgb_out


def makePlyFile(xyz, rgb, fileName='makeply.ply'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float) / 255.0)
    o3d.io.write_point_cloud(fileName, pcd)


def get_seg_coords_color(input_file):
    cloud = o3d.io.read_point_cloud(input_file)
    cloud = o3d.geometry.PointCloud.remove_non_finite_points(cloud, remove_nan = True, remove_infinite = False)
    cloud = o3d.geometry.PointCloud.remove_radius_outlier(cloud, 5, 0.025)[0]
    xyz = np.array(cloud.points)
    rgb = np.array(cloud.colors)*255
    return xyz, rgb


def get_boundry(xyz):
    xmin, xmax = xyz[0][0], xyz[0][0]
    ymin, ymax = xyz[0][1], xyz[0][1]
    zmin, zmax = xyz[0][2], xyz[0][2]
    for i in range(1, len(xyz)):
        if xmin > xyz[i][0]:
            xmin = xyz[i][0]
        if xmax < xyz[i][0]:
            xmax = xyz[i][0]
        if ymin > xyz[i][1]:
            ymin = xyz[i][1]
        if ymax < xyz[i][1]:
            ymax = xyz[i][1]
        if zmin > xyz[i][2]:
            zmin = xyz[i][2]
        if zmax < xyz[i][2]:
            zmax = xyz[i][2]
    return int(xmin*100)-5, int(xmax*100)+5, int(ymin*100)-5, int(ymax*100)+5, zmin, zmax


def makePNG(xyz, rgb, add_floor=True):
    # get image size
    xmin, xmax, ymin, ymax, zmin, zmax = get_boundry(xyz)
    w, h = xmax-xmin+1, ymax-ymin+1
    # pixelwise write
    ib = np.zeros((w,h),dtype=np.uint8)
    ig = np.zeros((w,h),dtype=np.uint8)
    ir = np.zeros((w,h),dtype=np.uint8)
    alpha = np.zeros((w,h),dtype=np.uint8)
    d = np.zeros((w,h),dtype=np.float32)
    for i in range(len(xyz)):
        if (rgb[i] == [143, 223, 142]).all(): # floor = np.array([143, 223, 142])
            continue
        x, y, z = xyz[i]
        if (rgb[i] == [171, 198, 230]).all() and z > zmax - 1: # wall = np.array([171, 198, 230])
            continue
        x = int(x*100) - xmin
        y = int(y*100) - ymax
        if [ir[x][y], ig[x][y], ib[x][y]] == [0,0,0]:
            # alpha[x][y] = 255
            for m in range(-1,2):
                for n in range(-1,2):
                    ir[x+m][y+n], ig[x+m][y+n], ib[x+m][y+n] = rgb[i]
                    alpha[x+m][y+n] = 255
                    d[x+m][y+n] = z
        # over write when the point is lower than the perious point
        elif d[x][y] > z and (rgb[i] != [171, 198, 230]).any():
            for m in range(-1,2):
                for n in range(-1,2):
                    ir[x+m][y+n], ig[x+m][y+n], ib[x+m][y+n] = rgb[i]
                    alpha[x+m][y+n] = 255
                    d[x+m][y+n] = z
    if add_floor:
        for i in range(len(xyz)):
            if (rgb[i] != [143, 223, 142]).all(): # floor = np.array([143, 223, 142])
                continue
            x, y, z = xyz[i]
            x = int(x*100) - xmin
            y = int(y*100) - ymax
            for m in range(-5,6):
                for n in range(-5,6):
                    if ir[x+m][y+n] == 0 and ig[x+m][y+n] == 0 and ib[x+m][y+n] == 0: # z < zmin + 1.5 and
                        ir[x+m][y+n], ig[x+m][y+n], ib[x+m][y+n] = rgb[i]
                        alpha[x+m][y+n] = 255
                        d[x+m][y+n] = z
    ir = Image.fromarray(ir)
    ig = Image.fromarray(ig)
    ib = Image.fromarray(ib)
    alpha = Image.fromarray(alpha)
    img = Image.merge('RGBA', (ir, ig, ib, alpha))
    return img
