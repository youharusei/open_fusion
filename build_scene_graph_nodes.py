import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import open3d as o3d
from openfusion.slam import build_slam, BaseSLAM
from openfusion.datasets import Dataset
from openfusion.utils import (
    show_pc, save_pc, get_cmap_legend
)
from configs.build import get_config

import torch
import pdb
from dataclasses import dataclass
import json
from dataclasses_json import dataclass_json
DBG = False
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="vlfusion", choices=["default", "cfusion", "vlfusion"])
    parser.add_argument('--vl', type=str, default="seem", help="vlfm to use")
    parser.add_argument('--data', type=str, default="kobuki", help='Path to dir of dataset.')
    parser.add_argument('--scene', type=str, default="icra", help='Name of the scene in the dataset.')
    parser.add_argument('--frames', type=int, default=-1, help='Total number of frames to use. If -1, use all frames.')
    parser.add_argument('--device_tsdf', type=str, default="cuda:0")
    parser.add_argument('--device_torch', type=str, default="cuda:1")
    parser.add_argument('--live', type=bool, default=False)
    parser.add_argument('--stream', type=bool, default=False)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--load', type=bool, default=True)
    parser.add_argument('--host_ip', type=str, default="127.0.0.1") # for stream
    args = parser.parse_args()

    params = get_config(args.data, args.scene)
    dataset:Dataset = params["dataset"](params["path"], args.frames)
    intrinsic = dataset.load_intrinsics(params["img_size"], params["input_size"])
    slam = build_slam(args, intrinsic, params)

    slam.point_state.load(f"{args.data}_{args.scene}/{args.algo}.npz")

    # load embedding dictionary
    emb_keys = slam.point_state.emb_keys
    emb_coords = slam.point_state.emb_coords
    emb_confs = slam.point_state.emb_confs
    emb_count = slam.point_state.emb_count
    emb_dict = slam.point_state.emb_dict
    num_obj_points_per_block : int = slam.point_state.num_obj_points_per_block
    buf_indices = slam.point_state.world.hashmap().active_buf_indices()
    buf_indices = torch.utils.dlpack.from_dlpack(buf_indices.to_dlpack())

    # extract valid area
    world_coords = emb_coords[buf_indices.cpu()].view(-1, num_obj_points_per_block, 3)
    mask_key = emb_keys[buf_indices.cpu()].view(-1, num_obj_points_per_block)
    valid_keys = torch.unique(mask_key)
    valid_embed = emb_dict[valid_keys]
    mask_pred_caption = valid_embed / (valid_embed.norm(dim=-1, keepdim=True) + 1e-7) # (E, D)
    BACK_GROUND = ["floor", "ground", "roof", "rooftree", "ceiling", "wall", "wallpaper"]
    embed_background = slam.vl_model.encode_prompt(BACK_GROUND, task="default")
    embed_background = embed_background / (embed_background.norm(dim=-1, keepdim=True) + 1e-7)
    
    # calculate the position and bounding box of every object
    object_centers = list()
    object_ranges = list()
    object_embeds = list()

    if DBG:
        points_world, colors_world = slam.point_state.get_pc()
    key_indice : int = 0 # the 0 object is always 0, skip
    for key_id in tqdm(valid_keys):
        # find the location of the key
        key_loc = torch.zeros_like(mask_key).bool()
        key_loc |= mask_key == key_id
        # find the points with the location of the key as indices
        points = world_coords[key_loc.unsqueeze(-1).repeat(1,1,3)].view(-1,3).numpy()
        mask_cls = torch.einsum("cd,nd->cn", mask_pred_caption[key_indice].unsqueeze(0).cuda(), embed_background)
        if points.shape[0] < 32 or mask_cls.max() > 0.15:
            key_indice += 1
            continue
        if DBG:
            points_world = np.concatenate([points_world, points], axis=0)
            colors_world = np.concatenate([colors_world, np.ones_like(points) * np.array([[255, 0, 0]])], axis=0)
            show_pc(points_world, colors_world)
        # calculate the center and bounding box of the object
        x_min = np.amin(points[:,0])
        x_max = np.amax(points[:,0])
        y_min = np.amin(points[:,1])
        y_max = np.amax(points[:,1])
        z_min = np.amin(points[:,2])
        z_max = np.amax(points[:,2])
        # NOTE: assume object_id to be same with the indice of object_*
        object_centers.append([(x_min + x_max)/2, (y_min + y_max)/2, (z_min + z_max)/2])
        object_ranges.append([x_max - x_min, y_max - y_min, z_max - z_min])
        object_embeds.append(valid_embed[key_indice].numpy())
        key_indice += 1
    object_id : int = 0
    object_nodes = []
    for object_embed in tqdm(object_embeds):
        object_node = {
            "object_id": object_id,
            "object_center": np.round(object_centers[object_id], 2).tolist(),
            "object_range": np.round(object_ranges[object_id], 2).tolist(),
            # "object_tag":
            # "object_caption":
        }
        object_nodes.append(object_node)
        object_id += 1
    with open(f"{args.data}_{args.scene}/scene_graph_nodes.json", "w") as f:
        json.dump(object_nodes, f, indent=4)
        f.close()
    np.savez(f"{args.data}_{args.scene}/object_embeds.npz", object_embeds = object_embeds)
    return

if __name__ == "__main__":
    main()