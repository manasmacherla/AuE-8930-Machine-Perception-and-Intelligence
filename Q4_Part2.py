import open3d as o3d
import numpy as np
import argparse

def arg_parse():
    """
        Command line arguments for different visualizations
        normal - For normal pcd visualization
        height - Coloring of points based on height
        internsity - Coloring of points based on intensity
    """

    parser = argparse.ArgumentParser(description = "Lidar argument parser")

    parser.add_argument("--type_color", dest = "type_color", default = "height", 
                        help = "Mention which type of colorization of points",
                        type = str)
    
    return parser.parse_args()

args = arg_parse()
type_color = args.type_color

pcd_name='data/sets/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151604048025.pcd.bin'
scan=np.fromfile(pcd_name, dtype=np.float32)
points = scan.reshape((-1, 5))[:, :4]

if type_color == "normal":
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    # pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])


if type_color == "height":
    ht_max = np.max(points[:,2]) #max height in the Z vector
    ht_min = np.min(points[:,2]) #min height in the Z vector

    color = np.zeros([len(points), 3])
    color[:, 0] = (points[:,2] - ht_min)/(ht_max - ht_min) # scaling values between 0 and 1
    color[:, 1] = 1 - (((points[:,2] - ht_min)/(ht_max - ht_min))) #scaling values between 0 and 1

    color[:,0] = 2*color[:,0]
    color[:,2] = 2*(1-color[:,2])

if type_color == "intensity":
    color = np.zeros([len(points), 3])
    color[:,0] = points[:,3]/255
    color[:,1] = (1 - points[:,3])/255

    color[:,0] = 2*color[:,0]
    color[:,2] = 2*(1-color[:,2])

if type_color == "label":
    seg_name='data/sets/nuscenes/lidarseg/v1.0-mini/0af1568c817a44048cfc67879f893f35_lidarseg.bin'
    seg=np.fromfile(seg_name, dtype=np.uint8)

    color = np.zeros([len(seg), 3])
    color[:, 0] = seg/32
    color[:, 1] = seg/32
    color[:, 2] = seg/32

color = color.astype(np.float64)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])