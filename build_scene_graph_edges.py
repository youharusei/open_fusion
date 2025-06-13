import json
import numpy as np
import torch
import pdb
import argparse
def calculate_3d_iou(box1:tuple, box2:tuple, padding = 0.0):
    """
    计算两个三维包围盒的交并比(IoU)
    参数:
        box1: (center, size) 例如 ((x1_c, y1_c, z1_c), (x1_s, y1_s, z1_s))
        box2: 格式同box1
        padding: 将包围盒扩大以检测靠近关系而非完全的相接关系
    返回:
        iou: 交并比值 [0, 1]
    """
    # 解包中心点和尺寸
    (c1, s1), (c2, s2) = box1, box2
    
    # 计算每个包围盒的min-max坐标
    min1 = [c - s/2 - padding for c, s in zip(c1, s1)]
    max1 = [c + s/2 + padding for c, s in zip(c1, s1)]
    min2 = [c - s/2 - padding for c, s in zip(c2, s2)]
    max2 = [c + s/2 + padding for c, s in zip(c2, s2)]
    
    # 计算交集区域的min-max坐标
    inter_min = [max(m1, m2) for m1, m2 in zip(min1, min2)]
    inter_max = [min(m1, m2) for m1, m2 in zip(max1, max2)]
    
    # 计算交集体积
    inter_size = [max(0, imax - imin) for imin, imax in zip(inter_min, inter_max)]
    inter_volume = inter_size[0] * inter_size[1] * inter_size[2]
    
    # 计算各自体积
    volume1 = s1[0] * s1[1] * s1[2]
    volume2 = s2[0] * s2[1] * s2[2]
    
    # 计算并集体积和IoU
    union_volume = volume1 + volume2 - inter_volume
    return inter_volume / union_volume if union_volume > 0 else 0.0

def calculate_embedding_similarity():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="kobuki", help='Path to dir of dataset.')
    parser.add_argument('--scene', type=str, default="icra", help='Name of the scene in the dataset.')
    args = parser.parse_args()

    with open(f"{args.data}_{args.scene}/scene_graph_nodes.json", "r") as f:
        object_nodes: list = json.load(f)
        f.close()
    n = object_nodes.__len__()
    graph_edges = []
    for i in range(n):
        for j in range(i+1, n):
            iou_ij = calculate_3d_iou(
                (object_nodes[i]["object_center"],object_nodes[i]["object_range"]),
                (object_nodes[j]["object_center"],object_nodes[j]["object_range"]))
            if iou_ij > 0.01:
                graph_edge = {
                    "related_nodes_id": [i, j],
                    "rough_relationship": f"object {i} is close to object {j}"
                }
                graph_edges.append(graph_edge)
    scene_graph = {
        "object_nodes": object_nodes,
        "graph_edges":graph_edges
    }
    with open(f"{args.data}_{args.scene}/scene_graph.json", "w") as f:
        json.dump(scene_graph, f, indent=2)
        f.close()

if __name__ == "__main__" :
    main()