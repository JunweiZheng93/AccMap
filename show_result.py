import numpy as np
import mayavi.mlab as mlab
from operator import itemgetter
import os, argparse
import glob
import open3d as o3d
import plyfile
import torch

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


def visualize_pts_rgb(fig, pts, rgb, scale=0.02):

    pxs = pts[:, 0]
    pys = pts[:, 1]
    pzs = pts[:, 2]
    pt_colors = np.zeros((pxs.size, 4), dtype=np.uint8)
    pt_colors[:, 0:3] = rgb
    pt_colors[:, 3] = 255  # transparent

    scalars = np.arange(pxs.__len__())
    points = mlab.points3d(pxs, pys, pzs,  scalars,
                           mode='sphere',  # point sphere
                           # colormap='Accent',
                           scale_mode='vector',
                           scale_factor=scale,
                           figure=fig)
    points.module_manager.scalar_lut_manager.lut.table = pt_colors

def get_coords_color(opt):
    input_file = os.path.join(opt.data_root, 'raw', opt.scene + '.ply')
    #input_file = os.path.join(opt.data_root, 'test', opt.scene + '_inst_nostuff.pth')
    assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)
    
    f = plyfile.PlyData().read(input_file)
    points = np.array([list(x) for x in f.elements[0]])
    xyz = np.ascontiguousarray(points[:, :3])
    rgb = np.ascontiguousarray(points[:, 3:6])

    #xyz, rgb = torch.load(input_file)
    return xyz, rgb
    

def get_semantic(opt):
    semantic_files = sorted(glob.glob(os.path.join(opt.result_root, 'semantic_npy', opt.scene + '*.npy')))
    rgb_out = None
    for semantic_file in semantic_files:
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(np.int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR)).astype(np.int)
        if np.shape(rgb_out) == ():
            rgb_out = label_pred_rgb
        else:
            rgb_out = np.concatenate((rgb_out,label_pred_rgb),axis=0)
    return rgb_out

def makePlyFile(xyz, rgb, fileName='makeply.ply'):
    # Make a ply file
    with open(fileName, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('comment PCL generated\n')
        f.write('element vertex {}\n'.format(len(xyz)))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyz)):
            r, g, b = rgb[i]
            x, y, z = xyz[i]
            f.write('{} {} {} {} {} {}\n'.format(x, y, z, r, g, b))
        print('Plyfile {} maked.'.format(fileName))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='path to the input dataset files', default='./testclouds/scans')
    parser.add_argument('--result_root', help='path to the predicted results', default='./testclouds/result')
    parser.add_argument('--scene', help='scene_name', default='06_01_r')
    parser.add_argument('--task', help='input or output', default='output')
    opt = parser.parse_args()

    print(opt.scene)

    xyz, rgb = get_coords_color(opt)
    rgb_out = get_semantic(opt)
    os.makedirs(os.path.join(opt.result_root, 'semantic_ply'), exist_ok=True)
    output_file = os.path.join(opt.result_root, 'semantic_ply', opt.scene + '.ply')
    if not os.path.isfile(output_file):
        makePlyFile(xyz, rgb_out, output_file)
    if opt.task == 'input':
        fig = mlab.figure(figure='Input', bgcolor=(1.0, 1.0, 1.0), size=((800, 800)))
        visualize_pts_rgb(fig, xyz, rgb)
    elif opt.task == 'output':
        fig = mlab.figure(figure='Output', bgcolor=(1.0, 1.0, 1.0), size=((800, 800)))
        visualize_pts_rgb(fig, xyz, rgb_out)
    mlab.show()


"""View point notes:
################# 06_01_r.ply
 `python show_result.py --scene 06_01_r --task output`

"""