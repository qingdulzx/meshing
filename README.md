
# 内容
## 对论文A hexahedral‑dominant FE meshing technique using trimmed hexahedral elements preserving sharp edges and corners的python代码实现
### 主要应用trimesh库
### 主要思路为
#### 1输入一个三维图形文件，将其以六面体网格划分
#### 2应用Marching Cubes算法对立方体网格做最初步的处理
#### 3针对图形特有的“锐边界”进行处理，修改已经得到的网格
#### 4进行拉普拉斯平滑，提升网格质量
