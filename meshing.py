import numpy as np  
import trimesh  
from skimage.measure import marching_cubes  
from scipy.spatial import ConvexHull  
  
# 定义网格类  
class Grid:  
    def __init__(self, vertices, faces):  
        self.vertices = vertices  
        self.faces = faces  
        self.edge_nodes = []  
        self.corner_nodes = []  
        self.cut_faces = [] 
  
    def add_edge_node(self, node):  
        if not np.any(np.isclose(self.vertices, node).all(axis=1)):  
            self.edge_nodes.append(node)  
  
    def add_corner_node(self, node):  
        if not np.any(np.isclose(self.vertices, node).all(axis=1)) and not np.any(np.isclose(self.edge_nodes, node).all(axis=1)):  
            self.corner_nodes.append(node)  
  
    def define_cut_faces(self):  
       new_vertices = []  
        for edge_node in self.edge_nodes:  
            closest_vertices_indices = self._find_closest_vertices_indices(edge_node)  
            for i in range(len(closest_vertices_indices)):  
                for j in range(i + 1, len(closest_vertices_indices)):  
                    face_indices = [closest_vertices_indices[i], closest_vertices_indices[j], len(self.vertices), len(self.vertices) + 1]  
                    new_vertices.extend([edge_node, edge_node])  
                    self.faces.append(face_indices)  
        new_vertices = np.unique(new_vertices, axis=0)  
        self.vertices = np.vstack((self.vertices, new_vertices))  
    def _find_closest_vertices_indices(self, point):   
        distances = np.linalg.norm(self.vertices - point, axis=1)  
        closest_indices = np.argsort(distances)[:3]  
        return closest_indices  
  
    def process_edge_nodes(self):  
        pass  
   def laplacian_smooth(self, iterations=1, alpha=0.5):  
        for _ in range(iterations):  
            new_vertices = []  
            for vertex in self.vertices:  
                adjacent_vertices = self._find_adjacent_vertices(vertex)  
                if len(adjacent_vertices) > 0:  
                    average_position = np.mean(adjacent_vertices, axis=0) * alpha + vertex * (1 - alpha)  
                    new_vertices.append(average_position)  
                else:  
                  
                    new_vertices.append(vertex)  
            self.vertices = np.array(new_vertices)  
   def _find_adjacent_vertices(self, vertex):  
        adjacent_vertices = []  
        for face in self.faces:  
            if vertex in self.vertices[face]:  
                adjacent_vertices.extend(self.vertices[face])  
        
        adjacent_vertices = np.unique(adjacent_vertices, axis=0)  
        adjacent_vertices = adjacent_vertices[~np.all(adjacent_vertices == vertex, axis=1)]  
        return adjacent_vertices  
    def process_corner_nodes(self):  
        if len(self.corner_nodes) > 0:  
            hull = ConvexHull(self.corner_nodes)  
            self.corner_nodes = self.corner_nodes[hull.vertices]  
def load_mesh(file_path):  
    return trimesh.load(file_path)  
  
 
def voxelize_mesh(mesh, voxel_size):  
    pitch = voxel_size  
    return mesh.voxelized(pitch=pitch)  
def apply_marching_cubes(voxel_data):  
    voxel_data = np.pad(voxel_data.matrix, 1, 'constant', constant_values=False)  
    vertices, faces, normals, values = marching_cubes(voxel_data)  
    return Grid(vertices, faces)  
 def extract_boundary_edges(mesh):  
    boundary_edges = []  
    for face in mesh.faces:  
        for i in range(3):  
            edge = sorted([face[i], face[(i + 1) % 3]])  
            if edge not in boundary_edges:  
                boundary_edges.append(edge)  
    return boundary_edges  
    pass  
  
def compute_edge_nodes(mesh, boundary_edges, voxel_size):  
    edge_nodes = []  
    for edge in boundary_edges:  
        start, end = mesh.vertices[edge]  
        line_segment = np.array([start, end])  
        for face in mesh.faces:  
            face_vertices = mesh.vertices[face]  
            if not np.any(np.isin(edge, face)):  
                # 计算线段和平面的交点  
                direction_vector = end - start  
                plane_normal = trimesh.triangles.normals(face_vertices)[0]  
                plane_point = np.mean(face_vertices, axis=0)   
                numerator = np.dot(plane_point - start, plane_normal)  
                denominator = np.dot(direction_vector, plane_normal)  
                if denominator != 0:  
                    t = numerator / denominator  
                    if 0 <= t <= 1:  
                        intersection = start + direction_vector * t  
                        edge_nodes.append(intersection)  
    edge_nodes = np.unique(np.round(edge_nodes, decimals=5), axis=0)  
    return edge_nodes  
  

def compute_corner_nodes(mesh, edge_nodes):  
    if len(edge_nodes) > 0:  
        hull = ConvexHull(edge_nodes)  
        return edge_nodes[hull.vertices].tolist()  
    else:  
        return []  
  
# 主处理函数  
def process_mesh(mesh, voxel_size):  
    voxels = voxelize_mesh(mesh, voxel_size)  
    grid = apply_marching_cubes(voxels)  
    boundary_edges = extract_boundary_edges(grid)  
    edge_nodes = compute_edge_nodes(grid, boundary_edges, voxel_size)  
    for node in edge_nodes:  
        grid.add_edge_node(node)  
    corner_nodes = compute_corner_nodes(grid, edge_nodes)  
    for node in corner_nodes:  
        grid.add_corner_node(node)  
    grid.define_cut_faces()  # 定义切割面  
    grid.process_edge_nodes()  # 处理边节点（可能需要根据具体需求进行修改）  
    grid.process_corner_nodes()  # 处理角节点  
    return grid  # 返回处理后的网格对象  
  
# 示例用法  
file_path = "path/to/your/model.stl"  # 替换为模型文件路径  
mesh = load_mesh(file_path)  
voxel_size = mesh.bounding_box.primitive.extents.min() / 100.0  
processed_grid = process_mesh(mesh, voxel_size)  
processed_grid.laplacian_smooth(iterations=5, alpha=0.5)  
