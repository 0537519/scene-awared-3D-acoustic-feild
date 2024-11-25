import torch
import trimesh
import numpy as np
from plyfile import PlyData, PlyElement
from options import Options
import os
import pickle

def search_nearest(coords_list, output_ply_path):
    vertices = []
    edges = []
    distances = []
    for i, coords in enumerate(coords_list):
        dist_vector = torch.norm(points - coords, dim=1)

        nearest_point_index = torch.argmin(dist_vector)
        min_distance = dist_vector[nearest_point_index].item()

        distances.append(min_distance)

        nearest_point = points[nearest_point_index].cpu().numpy()

        vertices.append((*coords.cpu().numpy(), 255, 0, 0))
        vertices.append((*nearest_point, 0, 255, 0))
        edges.append((2 * i, 2 * i + 1, 0, 0, 255))

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(coords_list)} points...")

    vertex_data = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    edge_data = np.array(edges, dtype=[('vertex1', 'i4'), ('vertex2', 'i4'),
                                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    edge_element = PlyElement.describe(edge_data, 'edge')
    PlyData([vertex_element, edge_element]).write(output_ply_path)
    print(f"All coords and their nearest points saved to {output_ply_path}")

    distance_tensor = torch.tensor(distances, device=device)
    return distance_tensor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ply_path = "D:/D/replica/replica_v1_0/room_2/habitat/mesh_frame_2.ply"
output_ply_path = "D:/D/replica/replica_v1_0/room_2/habitat/nearest_points_with_coords.ply"
output_ply_path_2 = "D:/D/replica/replica_v1_0/room_2/habitat/nearest_points_with_coords_2.ply"
points_path = "D:/D/LNAF/3/Learning_Neural_Acoustic_Fields-master/metadata/replica/room_2/points.txt"
mesh = trimesh.load(ply_path, process=True)
points = torch.tensor(mesh.vertices, device=device)

grid_gap = 0.25

cur_args = Options().parse()
minmax_base = cur_args.minmax_base
room_name = cur_args.apt

with open(os.path.join(minmax_base, room_name + "_minmax"), "rb") as min_max_loader:
    min_maxes = pickle.load(min_max_loader)
    min_pos = min_maxes[0][[0, 2]]
    max_pos = min_maxes[1][[0, 2]]

grid_coors_x = np.arange(min_pos[0], max_pos[0], grid_gap)
grid_coors_y = np.arange(min_pos[1], max_pos[1], grid_gap)

coords_list = []
z = -0.933901

for y in grid_coors_y:
    for x in grid_coors_x:
        coords = torch.tensor([float(x), -float(y), float(z)], device=device)
        coords_list.append(coords)

distance_tensor = search_nearest(coords_list, output_ply_path)

distance_file_path = "room_2_distance"
with open(distance_file_path, "wb") as f:
    pickle.dump(distance_tensor.cpu().numpy(), f)

print(f"Distance tensor saved to {distance_file_path}")

# Read and process points from points_path
coords_list_2 = []
with open(points_path, "r") as f:
    for line in f:
        data = line.strip().split()
        if len(data) == 4:
            x, y, z = float(data[1]), float(data[2]), float(data[3])
            coords = torch.tensor([x, y, z], device=device)
            coords_list_2.append(coords)

# Use search_nearest on points from points_path and save to output_ply_path_2
distance_tensor_2 = search_nearest(coords_list_2, output_ply_path_2)