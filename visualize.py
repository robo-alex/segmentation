import numpy as np
import yaml
import open3d
from pathlib import Path

 
points_dir = Path('/data3/zmt/dataset/SemanticKITTI/sequences/') # path to .bin data
label_dir = Path('/data3/zmt/dataset/SemanticKITTI/sequences/') # path to .label data

label_filter = [40, 48, 70, 72]    # object's label which you wan't to show
# label_filter = []
with open('/home/ou/workspace/code/Cylinder3D-master/config/label_mapping/semantic-kitti.yaml', 'r') as stream: # label_mapping configuration file
    label_mapping = yaml.safe_load(stream)
    color_dict = label_mapping['color_map']


def get_rgb_list(_label):
    c = color_dict[_label]
    return np.array((c[2], c[1], c[0]))


def draw_pc(pc_xyzrgb):
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
    pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)

    def custom_draw_geometry_with_key_callback(pcd):
        def change_background_to_black(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            opt.point_size = 1
            return False

        key_to_callback = {}
        key_to_callback[ord("K")] = change_background_to_black
        open3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

    custom_draw_geometry_with_key_callback(pc)


def concate_color(_points, _label):
    color = np.zeros((_points.shape[0], 3))
    label_id = np.unique(_label)
    for cls in label_id:
        if label_filter.__len__() == 0:
            color[_label == cls] = get_rgb_list(cls)
        elif label_filter.count(cls) == 0:
            color[_label == cls] = get_rgb_list(cls)
    _points = np.concatenate([_points, color], axis=1)
    return _points


for it in label_dir.iterdir():
    label_file = it
    points_file = points_dir / (str(it.stem) + '.bin')
    label = np.fromfile(label_file, dtype=np.uint32)
    points = np.fromfile(points_file, dtype=np.float32).reshape((-1, 4))[:, 0:3]
    print(label.shape, points.shape)
    colorful_points = concate_color(points, label)
    draw_pc(colorful_points)
