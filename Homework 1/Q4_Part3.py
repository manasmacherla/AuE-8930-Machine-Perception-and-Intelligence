import open3d as o3d
import numpy as np
from typing import Tuple, List, Dict
import struct
import argparse

def arg_parse():
    """
        Command line arguments for different visualizations
        normal - For normal pcd visualization
        height - Coloring of points based on height
        vx_comp - Coloring of points based on vx_comp
        vy_comp - Coloring of points based on vy_comp
    """


    parser = argparse.ArgumentParser(description = "Radar data argument parser")

    parser.add_argument("--type_color", dest = "type_color", default = "height", 
                        help = "Mention which type of colorization of points",
                        type = str)
    
    return parser.parse_args()

args = arg_parse()
type_color = args.type_color

# below function taken from the nuscenes dataset code
def from_file(file_name: str,invalid_states: List[int] = None,dynprop_states: List[int] = None, ambig_states: List[int] = None):
    """
    Loads RADAR data from a Point Cloud Data file. See details below.
    :param file_name: The path of the pointcloud file.
    :param invalid_states: Radar states to be kept. See details below.
    :param dynprop_states: Radar states to be kept. Use [0, 2, 6] for moving objects only. See details below.
    :param ambig_states: Radar states to be kept. See details below.
    To keep all radar returns, set each state filter to range(18).
    :return: <np.float: d, n>. Point cloud matrix with d dimensions and n points.
    Example of the header fields:
    # .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
    SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
    TYPE F F F I I F F F F F I I I I I I I I
    COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    WIDTH 125
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS 125
    DATA binary
    Below some of the fields are explained in more detail:
    x is front, y is left
    vx, vy are the velocities in m/s.
    vx_comp, vy_comp are the velocities in m/s compensated by the ego motion.
    We recommend using the compensated velocities.
    invalid_state: state of Cluster validity state.
    (Invalid states)
    0x01	invalid due to low RCS
    0x02	invalid due to near-field artefact
    0x03	invalid far range cluster because not confirmed in near range
    0x05	reserved
    0x06	invalid cluster due to high mirror probability
    0x07	Invalid cluster because outside sensor field of view
    0x0d	reserved
    0x0e	invalid cluster because it is a harmonics
    (Valid states)
    0x00	valid
    0x04	valid cluster with low RCS
    0x08	valid cluster with azimuth correction due to elevation
    0x09	valid cluster with high child probability
    0x0a	valid cluster with high probability of being a 50 deg artefact
    0x0b	valid cluster but no local maximum
    0x0c	valid cluster with high artefact probability
    0x0f	valid cluster with above 95m in near range
    0x10	valid cluster with high multi-target probability
    0x11	valid cluster with suspicious angle
    dynProp: Dynamic property of cluster to indicate if is moving or not.
    0: moving
    1: stationary
    2: oncoming
    3: stationary candidate
    4: unknown
    5: crossing stationary
    6: crossing moving
    7: stopped
    ambig_state: State of Doppler (radial velocity) ambiguity solution.
    0: invalid
    1: ambiguous
    2: staggered ramp
    3: unambiguous
    4: stationary candidates
    pdh0: False alarm probability of cluster (i.e. probability of being an artefact caused by multipath or similar).
    0: invalid
    1: <25%
    2: 50%
    3: 75%
    4: 90%
    5: 99%
    6: 99.9%
    7: <=100%
    """

    assert file_name.endswith('.pcd'), 'Unsupported filetype {}'.format(file_name)

    meta = []
    with open(file_name, 'rb') as f:
        for line in f:
            line = line.strip().decode('utf-8')
            meta.append(line)
            if line.startswith('DATA'):
                break

        data_binary = f.read()

    # Get the header rows and check if they appear as expected.
    assert meta[0].startswith('#'), 'First line must be comment'
    assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
    sizes = meta[3].split(' ')[1:]
    types = meta[4].split(' ')[1:]
    counts = meta[5].split(' ')[1:]
    width = int(meta[6].split(' ')[1])
    height = int(meta[7].split(' ')[1])
    data = meta[10].split(' ')[1]
    feature_count = len(types)
    assert width > 0
    assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
    assert height == 1, 'Error: height != 0 not supported!'
    assert data == 'binary'

    # Lookup table for how to decode the binaries.
    unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                        'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                        'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
    types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

    # Decode each point.
    offset = 0
    point_count = width
    points = []
    for i in range(point_count):
        point = []
        for p in range(feature_count):
            start_p = offset
            end_p = start_p + int(sizes[p])
            assert end_p < len(data_binary)
            point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
            point.append(point_p)
            offset = end_p
        points.append(point)

    # A NaN in the first point indicates an empty pointcloud.
    point = np.array(points[0])
    if np.any(np.isnan(point)):
        return cls(np.zeros((feature_count, 0)))

    # Convert to numpy matrix.
    points = np.array(points).transpose()

    return points

pcd_name='data/sets/nuscenes/samples/RADAR_FRONT/n008-2018-08-01-15-16-36-0400__RADAR_FRONT__1533151619475551.pcd'

radar_array = from_file(pcd_name)
radar_array = np.transpose(radar_array)

if type_color == "normal":
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(radar_array[:, :3])
    o3d.visualization.draw_geometries([pcd])

if type_color == "height":
    z_max = np.max(radar_array[:,2]) + 0.001 # adding the value for numerical stability, 8th index value is vx_comp, 2nd value is height
    z_min = np.min(radar_array[:,2]) + 0.001

    print(radar_array[:,2])

    color = np.zeros([radar_array.shape[0], 3])
    color[:,0] = (radar_array[:,2] - z_min)/(z_max - z_min)
    color[:,2] = (radar_array[:,2] - z_min)/(z_max - z_min)

    color[:,0] = 2*color[:,0]
    color[:,2] = 2*(1-color[:,2])

if type_color == "vx_comp":
    vx_max = np.max(radar_array[:,8]) + 0.001 # adding the value for numerical stability, 8th index value is vx_comp, 2nd value is height
    vx_min = np.min(radar_array[:,8]) + 0.001

    color = np.zeros([radar_array.shape[0], 3])
    color[:,0] = (radar_array[:,8] - vx_min)/(vx_max - vx_min)
    color[:,2] = (radar_array[:,8] - vx_min)/(vx_max - vx_min)

    color[:,0] = 2*color[:,0]
    color[:,2] = 2*(1-color[:,2])

if type_color == "vy_comp":
    vy_max = np.max(radar_array[:,9]) + 0.001 # adding the value for numerical stability, 8th index value is vx_comp, 2nd value is height
    vy_min = np.min(radar_array[:,9]) + 0.001

    color = np.zeros([radar_array.shape[0], 3])
    color[:,0] = (radar_array[:,9] - vy_min)/(vy_max - vy_min)
    color[:,2] = (radar_array[:,9] - vy_min)/(vy_max - vy_min)

    color[:,0] = 2*color[:,0]
    color[:,2] = 2*(1-color[:,2])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(radar_array[:, :3])
pcd.colors = o3d.utility.Vector3dVector(color)
o3d.visualization.draw_geometries([pcd])