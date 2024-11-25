frame_list=[40,93,31]
import json
import numpy as np
from plyfile import PlyData, PlyElement

ply_path = "D:/D/replica/replica_v1_0/apartment_2/habitat/mesh_semantic.ply"
ply_path_frame = "D:/D/replica/replica_v1_0/apartment_2/habitat/mesh_frame_2.ply"

json_file_path = "D:/D/replica/replica_v1_0/apartment_2/habitat/info_semantic.json"

with open(json_file_path, 'r') as file:
    data = json.load(file)

id_to_label = data.get("id_to_label", [])
object_id_to_label = {object_id: label for object_id, label in enumerate(id_to_label)}

with open(ply_path, 'rb') as f:
    plydata = PlyData.read(f)

vertices = plydata['vertex']
faces = plydata['face']

filtered_faces = []
vertex_indices_in_faces = set()

for face in faces.data:
    object_id = face['object_id']
    label = object_id_to_label.get(object_id, None)
    if label in frame_list:
        filtered_faces.append(face)
        vertex_indices_in_faces.update(face['vertex_indices'])

filtered_vertices = []
filtered_vertex_indices = {}

for index, vertex in enumerate(vertices.data):
    if index in vertex_indices_in_faces:
        filtered_vertices.append(vertex)
        filtered_vertex_indices[index] = len(filtered_vertices) - 1

filtered_vertices_array = np.array(filtered_vertices)

filtered_faces_data = []
for face in filtered_faces:
    new_vertex_indices = [filtered_vertex_indices[index] for index in face['vertex_indices'] if index in filtered_vertex_indices]
    filtered_faces_data.append((new_vertex_indices, face['object_id']))

filtered_faces_array = np.array(
    [(np.array(face[0]), face[1]) for face in filtered_faces_data],
    dtype=[('vertex_indices', 'i4', (len(filtered_faces_data[0][0]),)), ('object_id', 'i4')]
)

new_vertices_element = PlyElement.describe(filtered_vertices_array, 'vertex')
new_faces_element = PlyElement.describe(filtered_faces_array, 'face')

new_plydata = PlyData([new_vertices_element, new_faces_element], text=True)
with open(ply_path_frame, 'wb') as f_out:
    new_plydata.write(f_out)

print("new ply file created.")