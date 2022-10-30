import open3d as o3d, os, argparse
import numpy as np
def read_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    print(pcd)
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = [1,0,0]
    o3d.visualization.draw_geometries([pcd], zoom=0.5, front=[-0.03, 0.14, 0.99],  lookat=[4.27, 6.47, -7.32], up=[0.86, -0.49, 0.09])
# "{ "class_name" : "ViewTrajectory", "interval" : 29, "is_loop" : false, "trajectory" : [ { "boundingbox_max" : [ 3.0043845176696777, 3.4275100231170654, 1.0038625001907349 ], "boundingbox_min" : [ -2.8541419506072998, -7.3357467651367188, -2.7678806781768799 ], "field_of_view" : 60.0, "front" : [ 0.14618647666259943, 0.024620471521868813, 0.98895062891077434 ], "lookat" : [ 0.075121283531188965, -1.9541183710098267, -0.88200908899307251 ], "up" : [ -0.989210656699131, -0.0060436953197784927, 0.14637537504565365 ], "zoom" : 0.49999999999999978 } ], "version_major" : 1, "version_minor" : 0 }
# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" :
# 	[
# 		{
# 			"boundingbox_max" : [ 15.603017807006836, 22.644699096679688, 1.4711111783981323 ],
# 			"boundingbox_min" : [ -8.4470682144165039, -7.4007821083068848, -6.0798811912536621 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ -0.030701388848682516, 0.14222177533478988, 0.98935857571629848 ],
# 			"lookat" : [ 4.2729048999008459, 6.4771201754002856, -7.3208528301063689 ],
# 			"up" : [ 0.86484114723152017, -0.49246144447802148, 0.097629482011780205 ],
# 			"zoom" : 0.43999999999999995
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to the data files', default='./testclouds/result/semantic_ply')
    parser.add_argument('--scene', help='scene_no.', default='06_01_r')
    opt = parser.parse_args()
    file_path = os.path.join(opt.path, opt.scene+ '.ply')
    print('Loading point clouds:{}'.format(file_path))
    # read_pcd('/home/cvhci/Downloads/plydata/ply_xyz_rgb_update_16_05/04_04_r.ply')
    # --- multiple areas of CVHCI
    # read_pcd('/home/cvhci/Downloads/plydata/ply_xyz_semantic_update_16_05/04_04_r.ply')
    # --- kitchen room of CVHCI
    # read_pcd('/home/cvhci/Downloads/plydata/ply_xyz_semantic_update_16_05/05_02_r.ply')

    # --- meeting room of CVHCI
    # View point
    # {
    # 	"class_name" : "ViewTrajectory",
    # 	"interval" : 29,
    # 	"is_loop" : false,
    # 	"trajectory" :
    # 	[
    # 		{
    # 			"boundingbox_max" : [ 10.284750938415527, 8.4878435134887695, 1.1389248371124268 ],
    # 			"boundingbox_min" : [ -4.5652732849121094, -1.4953155517578125, -6.033604621887207 ],
    # 			"field_of_view" : 60.0,
    # 			"front" : [ 0.36325299510280157, -0.031652138130757297, 0.93115272845038455 ],
    # 			"lookat" : [ 0.15173084997682248, 2.0439083138741974, -7.1795401241657109 ],
    # 			"up" : [ -0.93156970984277654, -0.028434501403992139, 0.36244910654235507 ],
    # 			"zoom" : 0.60000000000000009
    # 		}
    # 	],
    # 	"version_major" : 1,
    # 	"version_minor" : 0
    # }
    read_pcd(file_path)
    # read_pcd('/home/cvhci/Downloads/plydata/ply_xyz_semantic_update_16_05/06_01_r.ply')

