import numpy as np
from operator import itemgetter
import os, argparse
import glob
import open3d as o3d
import plyfile
import cv2

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

def get_coords_color(points):
    input_file = os.path.join(opt.data_root, opt.scene + '.ply')
    # input_file = os.path.join(opt.data_root, 'test', opt.scene + '_inst_nostuff.pth')
    assert os.path.isfile(input_file), 'File not exist - {}.'.format(input_file)
    print('Removing outliers: {}.ply ...'.format(opt.scene))
    cloud = o3d.io.read_point_cloud(input_file)
    cloud = o3d.geometry.PointCloud.remove_non_finite_points(cloud, remove_nan = True, remove_infinite = False)
    cloud = o3d.geometry.PointCloud.remove_radius_outlier(cloud, 5, 0.025)[0]

    print('Readind data: {}.ply ...'.format(opt.scene))
    #o3d.visualization.draw_geometries([cloud])
    xyz = np.array(cloud.points)
    rgb = np.array(cloud.colors)*255

    return xyz, rgb

def get_boundry(xyz):
    print('get bounday from point cloud shape:{}'.format(xyz.shape))
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
    print('image size {} x {}'.format(w,h))
    # pixelwise write
    ib = np.zeros((w,h),dtype=np.uint8)
    ig = np.zeros((w,h),dtype=np.uint8)
    ir = np.zeros((w,h),dtype=np.uint8)
    alpha = np.zeros((w,h),dtype=np.uint8)
    d = np.zeros((w,h),dtype=np.float32)
    print("Adding Objects ...")
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
            # ir[x][y], ig[x][y], ib[x][y] = rgb[i]
            # d[x][y] = z
        # over write when the point is lower than the perious point
        elif d[x][y] > z and (rgb[i] != [171, 198, 230]).any():
            for m in range(-1,2):
                for n in range(-1,2):
                    ir[x+m][y+n], ig[x+m][y+n], ib[x+m][y+n] = rgb[i]
                    alpha[x+m][y+n] = 255
                    d[x+m][y+n] = z
            # ir[x][y], ig[x][y], ib[x][y] = rgb[i]
            # d[x][y] = z
    if add_floor:
        print("Adding floor ...")
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
    print("Generating image ...")
    img = cv2.merge([ib, ig, ir, alpha])
    # kernel = np.ones((3, 3), dtype=np.uint8)
    # img = cv2.dilate(img, kernel, 1)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_root', help='path to the 3d semantic point clouds files', default='./plyfiles/results')
    parser.add_argument('--data_root', help='path to the 3d semantic point clouds files',
                        default='testclouds/result/semantic_ply')
    parser.add_argument('--scene', help='scene_name', default='06_01_r')
    opt = parser.parse_args()
    xyz, rgb = get_coords_color(opt)
    print('Generating image for scene:{} ...'.format(opt.scene))
    add_floor = True
    image = makePNG(xyz, rgb, add_floor)
    if add_floor:
        cv2.imwrite('{}_floor.png'.format(opt.scene),image)
    else:
        cv2.imwrite('{}.png'.format(opt.scene),image)
    print('Generation 2D-Map for {} done!'.format(opt.scene))