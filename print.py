import numpy as np
from plyfile import PlyData
import time

EPSILON = 1e-8

class Intersection:
    def __init__(self):
        self.happened = False
        self.coords = None
        self.normal = None
        self.distance = float('inf')
        self.obj = None
        self.back=False

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = np.array(direction)
        self.direction_inv = np.where(self.direction != 0, 1.0 / self.direction, np.inf)
        self.dir_is_neg = self.direction_inv < 0

    def __call__(self, t):
        return self.origin + t * self.direction

class Triangle:
    def __init__(self, v0, v1, v2, normal):
        self.v0 = np.array(v0)
        self.v1=np.array(v1)
        self.v2=np.array(v2)
        self.e1 = np.array(v1) - np.array(v0)
        self.e2 = np.array(v2) - np.array(v0)
        self.normal = np.array(normal)

    def get_intersection(self, ray):
        inter = Intersection()

        if np.dot(ray.direction, self.normal) > 0:
           inter.back=True

        pvec = np.cross(ray.direction, self.e2)
        det = np.dot(self.e1, pvec)

        if abs(det) < EPSILON:
            return inter

        det_inv = 1.0 / det
        tvec = ray.origin - self.v0
        u = np.dot(tvec, pvec) * det_inv

        if u < 0 or u > 1:
            return inter

        qvec = np.cross(tvec, self.e1)
        v = np.dot(ray.direction, qvec) * det_inv

        if v < 0 or u + v > 1:
            return inter

        t_tmp = np.dot(self.e2, qvec) * det_inv

        if t_tmp < 0:
            return inter

        inter.happened = True
        inter.coords = ray(t_tmp)
        inter.normal = self.normal
        inter.distance = t_tmp
        inter.obj = self

        return inter

    def get_bounds(self):
        min_x = min(self.v0[0], self.v1[0], self.v2[0])
        min_y = min(self.v0[1], self.v1[1], self.v2[1])
        min_z = min(self.v0[2], self.v1[2], self.v2[2])
        max_x = max(self.v0[0], self.v1[0], self.v2[0])
        max_y = max(self.v0[1], self.v1[1], self.v2[1])
        max_z = max(self.v0[2], self.v1[2], self.v2[2])
        return Bounds3(p1=[min_x, min_y, min_z], p2=[max_x, max_y, max_z])

class Quadrilateral:
    def __init__(self, v0, v1, v2, v3, normal):
        self.v0=np.array(v0)
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.v3 = np.array(v3)
        self.triangle1 = Triangle(v0, v1, v2, normal)
        self.triangle2 = Triangle(v2, v3, v0, normal)

    def get_intersection(self, ray):
        inter1 = self.triangle1.get_intersection(ray)
        if inter1.happened:
            return inter1

        inter2 = self.triangle2.get_intersection(ray)
        return inter2

    def get_bounds(self):
        min_x = min(self.v0[0], self.v1[0], self.v2[0], self.v3[0])
        min_y = min(self.v0[1], self.v1[1], self.v2[1], self.v3[1])
        min_z = min(self.v0[2], self.v1[2], self.v2[2], self.v3[2])
        max_x = max(self.v0[0], self.v1[0], self.v2[0], self.v3[0])
        max_y = max(self.v0[1], self.v1[1], self.v2[1], self.v3[1])
        max_z = max(self.v0[2], self.v1[2], self.v2[2], self.v3[2])

        return Bounds3(p1=[min_x, min_y, min_z], p2=[max_x, max_y, max_z])


class Bounds3:
    def __init__(self, p1=None, p2=None):
        if p1 is None and p2 is None:
            min_num = -np.inf
            max_num = np.inf
            self.pMin = np.array([max_num, max_num, max_num])
            self.pMax = np.array([min_num, min_num, min_num])
        elif p2 is None:
            self.pMin = np.array(p1)
            self.pMax = np.array(p1)
        else:
            self.pMin = np.array([min(p1[0], p2[0]), min(p1[1], p2[1]), min(p1[2], p2[2])])
            self.pMax = np.array([max(p1[0], p2[0]), max(p1[1], p2[1]), max(p1[2], p2[2])])

    def intersect_p(self, ray):
        tEnter = -np.inf
        tExit = np.inf
        for i in range(3):
            if ray.direction[i] != 0:
                tMin = (self.pMin[i] - ray.origin[i]) * ray.direction_inv[i]
                tMax = (self.pMax[i] - ray.origin[i]) * ray.direction_inv[i]
                if ray.dir_is_neg[i]:
                    tMin, tMax = tMax, tMin
            else:
                if ray.origin[i] < self.pMin[i] or ray.origin[i] > self.pMax[i]:
                    return False
                tMin = -np.inf
                tMax = np.inf
            tEnter = max(tEnter, tMin)
            tExit = min(tExit, tMax)
        return tEnter <= tExit and tExit >= 0

    def centroid(self):
        return (self.pMin + self.pMax) / 2.0

    def max_extent(self):
        extent = self.pMax - self.pMin
        return np.argmax(extent)

    def surface_area(self):
        width = self.pMax[0] - self.pMin[0]
        height = self.pMax[1] - self.pMin[1]
        depth = self.pMax[2] - self.pMin[2]
        return 2 * (width * height + height * depth + depth * width)

class BVHBuildNode:
    def __init__(self):
        self.bounds = Bounds3()
        self.left = None
        self.right = None
        self.object = None
        self.splitAxis = 0
        self.firstPrimOffset = 0
        self.nPrimitives = 0


def Union(bounds1, bounds2):
    pMin = np.minimum(bounds1.pMin, bounds2.pMin)
    pMax = np.maximum(bounds1.pMax, bounds2.pMax)

    return Bounds3(pMin, pMax)

class BVHAccel:
    def get_intersection(self, node, ray):
        inter = Intersection()

        if not node.bounds.intersect_p(ray):
            return inter

        if node.left is None and node.right is None:
            return node.object.get_intersection(ray)

        l = None
        if node.left is not None:
            l = self.get_intersection(node.left, ray)

        r = None
        if node.right is not None:
            r = self.get_intersection(node.right, ray)

        if l is None and r is None:
            return None
        elif l is None:
            return r
        elif r is None:
            return l
        else:
            return l if l.distance < r.distance else r

    def recursive_build(self, objects,ray):
        node = BVHBuildNode()
        bounds = Bounds3()
        for obj in objects:
            bounds = Union(bounds, obj.get_bounds())
        if not bounds.intersect_p(ray):
            return None

        total_objects = len(objects)
        if(total_objects>10000):
          print(f"Building BVH for {total_objects} objects...")

        if len(objects) == 1:
            node.bounds = objects[0].get_bounds()
            node.object = objects[0]
            node.left = None
            node.right = None
            return node
        elif len(objects) == 2:
            node.left = self.recursive_build([objects[0]],ray)
            node.right = self.recursive_build([objects[1]],ray)
            if node.left is not None and node.right is not None:
                node.bounds = Union(node.left.bounds, node.right.bounds)
            elif node.left is not None:
                node.bounds = Union(node.left.bounds, bounds)
            elif node.right is not None:
                node.bounds = Union(bounds, node.right.bounds)
            return node

        else:
            centroid_bounds = Bounds3()
            for obj in objects:
                centroid = obj.get_bounds().centroid()
                # Wrap the centroid in a Bounds3 object
                centroid_bounds = Union(centroid_bounds, Bounds3(centroid, centroid))

            dim = centroid_bounds.max_extent()
            if dim == 0:
                objects.sort(key=lambda f: f.get_bounds().centroid()[0])  # Use indexing for numpy array
            elif dim == 1:
                objects.sort(key=lambda f: f.get_bounds().centroid()[1])
            elif dim == 2:
                objects.sort(key=lambda f: f.get_bounds().centroid()[2])

            beginning = 0
            middling = len(objects) // 2
            ending = len(objects)

            left_shapes = objects[beginning:middling]
            right_shapes = objects[middling:ending]

            assert len(objects) == (len(left_shapes) + len(right_shapes))

            node.left = self.recursive_build(left_shapes,ray)
            node.right = self.recursive_build(right_shapes,ray)
            if node.left is not None and node.right is not None:
                node.bounds = Union(node.left.bounds, node.right.bounds)
            elif node.left is not None:
                node.bounds = Union(node.left.bounds, bounds)
            elif node.right is not None:
                node.bounds = Union(bounds, node.right.bounds)

        return node

def ray_intersect(ray):
    objects = []
    for i, face in enumerate(face_data['vertex_indices']):
        v0_idx, v1_idx, v2_idx, v3_idx = face
        v0 = vertex_data[v0_idx]
        v1 = vertex_data[v1_idx]
        v2 = vertex_data[v2_idx]
        v3 = vertex_data[v3_idx]

        v0_coords = np.array([v0['x'], v0['y'], v0['z']])
        v0_normal = np.array([v0['nx'], v0['ny'], v0['nz']])

        v1_coords = np.array([v1['x'], v1['y'], v1['z']])
        v1_normal = np.array([v1['nx'], v1['ny'], v1['nz']])

        v2_coords = np.array([v2['x'], v2['y'], v2['z']])
        v2_normal = np.array([v2['nx'], v2['ny'], v2['nz']])

        v3_coords = np.array([v3['x'], v3['y'], v3['z']])
        v3_normal = np.array([v3['nx'], v3['ny'], v3['nz']])

        n=((v0_normal+v1_normal+v2_normal+v3_normal)/4)

        quadrilateral = Quadrilateral(v0=v0_coords, v1=v1_coords, v2=v2_coords,
                                   v3=v3_coords, normal=n)
        objects.append(quadrilateral)

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1} faces...")

    bvh_accel = BVHAccel()
    centroid_bounds = Bounds3()
    for obj in objects:
        bounds = obj.get_bounds()
        centroid_bounds = Union(centroid_bounds, bounds)

    bvh_root = bvh_accel.recursive_build(objects,ray)

    # Check intersection
    intersection_result = bvh_accel.get_intersection(bvh_root, ray)

    if intersection_result:
        print(f"Intersection at distance: {intersection_result.distance}")
    else:
        print("No intersection found.")

    return intersection_result

if __name__ == "__main__":
    start_time = time.time()

    ply_path = "D:/D/replica/replica_v1_0/apartment_1/habitat/mesh_frame_2.ply"
    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data['vertex']
    face_data = ply_data['face']

    num_rays = 6
    ray_origin = np.array([0.7, -0.5, -0.8])
    intersections_coords = []
    intersections_dist = []

    ray_direction_x_positive = np.array([1, 0, 0])
    ray_x_positive = Ray(ray_origin, ray_direction_x_positive)
    intersection_x_positive = ray_intersect(ray_x_positive)
    intersections_coords.append(intersection_x_positive.coords)
    intersections_dist.append(intersection_x_positive.distance)
    print("ray num 1")

    ray_direction_x_negative = np.array([-1, 0, 0])
    ray_x_negative = Ray(ray_origin, ray_direction_x_negative)
    intersection_x_negative = ray_intersect(ray_x_negative)
    intersections_coords.append(intersection_x_negative.coords)
    intersections_dist.append(intersection_x_negative.distance)
    print("ray num 2")

    ray_direction_y_positive = np.array([0, 1, 0])
    ray_y_positive = Ray(ray_origin, ray_direction_y_positive)
    intersection_y_positive = ray_intersect(ray_y_positive)
    intersections_coords.append(intersection_y_positive.coords)
    intersections_dist.append(intersection_y_positive.distance)
    print("ray num 3")

    ray_direction_y_negative = np.array([0, -1, 0])
    ray_y_negative = Ray(ray_origin, ray_direction_y_negative)
    intersection_y_negative = ray_intersect(ray_y_negative)
    intersections_coords.append(intersection_y_negative.coords)
    intersections_dist.append(intersection_y_negative.distance)
    print("ray num 4")

    ray_direction_z_positive = np.array([0, 0, 1])
    ray_z_positive = Ray(ray_origin, ray_direction_z_positive)
    intersection_z_positive = ray_intersect(ray_z_positive)
    intersections_coords.append(intersection_z_positive.coords)
    intersections_dist.append(intersection_z_positive.distance)
    print("ray num 5")

    ray_direction_z_negative = np.array([0, 0, -1])
    ray_z_negative = Ray(ray_origin, ray_direction_z_negative)
    intersection_z_negative = ray_intersect(ray_z_negative)
    intersections_coords.append(intersection_z_negative.coords)
    intersections_dist.append(intersection_z_negative.distance)
    print("ray num 6")

    min_distance_index = intersections_dist.index(min(intersections_dist))
    print("coord:" + str(intersections_coords[min_distance_index]))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"running time: {elapsed_time:.2f} s")
