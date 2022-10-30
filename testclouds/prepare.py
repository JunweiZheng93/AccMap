import glob
import plyfile
import os

import numpy as np
import multiprocessing as mp
import torch, argparse
# import open3d as o3d

def f_test(fn):
    #if fn[-3:] == 'pcd':
    #    pcd = o3d.io.read_point_cloud(fn)
    #    o3d.io.write_point_cloud(fn[:-4]+".ply", pcd, compressed=True)
    #    fn = fn[:-4]+".ply"
    f = plyfile.PlyData().read(fn)
    # x,y,z,r,g,b,nx,ny,nz,curvature
    points = np.array([list(x) for x in f.elements[0]], dtype=np.float32)
    coords = np.ascontiguousarray(points[:, :3] - points[:, :3].mean(0))
    colors = np.ascontiguousarray(points[:, 3:6]) / 127.5 - 1
    #print(colors)
    # --- from raw ply file to torch format
    target_fn = os.path.join(target, os.path.basename(fn).replace('.ply', '_inst_nostuff.pth'))
    torch.save((coords, colors), target_fn)
    print('Saving to ', target_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='path to raw point clouds', default='./testclouds/scans/raw')
    parser.add_argument('--data_name', help='process a single file', default=None)
    parser.add_argument('--data_output', help='output path', default='./testclouds/scans/test')
    opt = parser.parse_args()
    if opt.data_name != None:
        files = sorted(glob.glob(opt.data_root + '/' + opt.data_name + '.ply')) # + sorted(glob.glob(opt.data_root + '/*.pcd'))
    else:
        files = sorted(glob.glob(opt.data_root + '/*.ply')) # + sorted(glob.glob(opt.data_root + '/*.pcd'))
    global target 
    target = opt.data_output
    os.makedirs(target, exist_ok=True)
    p = mp.Pool(processes=mp.cpu_count())
    p.map(f_test, files)
    p.close()
    p.join()