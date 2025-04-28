""" 
ROS node of goal location query server.
load the tsdf and embedding dict in the manner of main.py,
setup the server to query the location of top-1 instance with semantic query

service: goal_location_query
request:
    string semantic query
response:
    bool success
    float location x
    float location y
    float size range_x
    float size range_y
"""
#! /home/ycs/.conda/envs/open_fusion/bin/python
import sys
sys.path.append('~/catkin_ws/src/goal_location_query/src')
import argparse
import os
import time
import numpy as np
from tqdm import tqdm
import open3d as o3d
from openfusion.slam import build_slam, BaseSLAM
from openfusion.datasets import Dataset
from configs.build import get_config

import rospy
from goal_location_query.srv import GoalLocationQuery, GoalLocationQueryRequest, GoalLocationQueryResponse

class LocationQueryServerROS(object):
    def __init__(self, slam:BaseSLAM):
        self.slam = slam
        self.service = rospy.Service("goal_location_query", GoalLocationQuery, self.goal_location_query_callback)
    
    def goal_location_query_callback(self, request:GoalLocationQueryRequest):
        points = self.slam.fast_query(query=request.sematic_query, only_poi=True, topk=1, n_points=-1)
        if points.size == 0: return GoalLocationQueryResponse(False, 0, 0, 0, 0)
        x_min = np.amin(points[:,0])
        x_max = np.amax(points[:,0])
        y_min = np.amin(points[:,1])
        y_max = np.amax(points[:,1])
        x = (x_min + x_max)/2
        y = (y_min + y_max)/2
        range_x = x_max - x_min
        range_y = y_max - y_min
        return GoalLocationQueryResponse(True, x, y, range_x, range_y)

def main():
    rospy.init_node("goal_location_server_node")
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default="vlfusion", choices=["default", "cfusion", "vlfusion"])
    parser.add_argument('--vl', type=str, default="seem", help="vlfm to use")
    parser.add_argument('--data', type=str, default="rgbd", help='Path to dir of dataset.')
    parser.add_argument('--scene', type=str, default="wuhu_1", help='Name of the scene in the dataset.')
    parser.add_argument('--frames', type=int, default=-1, help='Total number of frames to use. If -1, use all frames.')
    parser.add_argument('--device_tsdf', type=str, default="cuda:0")
    parser.add_argument('--device_torch', type=str, default="cuda:0")
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--stream', action='store_true')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--load', type=bool, default=True)
    parser.add_argument('--host_ip', type=str, default="YOUR IP") # for stream
    args = parser.parse_args()

    params = get_config(args.data, args.scene)
    dataset:Dataset = params["dataset"](params["path"], args.frames, args.stream)
    intrinsic = dataset.load_intrinsics(params["img_size"], params["input_size"])
    slam = build_slam(args, intrinsic, params)

    if os.path.exists(f"{args.data}_{args.scene}/{args.algo}.npz"):
        rospy.loginfo("[*] loading saved state...")
        slam.point_state.load(f"{args.data}_{args.scene}/{args.algo}.npz")
    else:
        rospy.loginfo("[*] no saved state found!")
        return

    server = LocationQueryServerROS(slam)
    rospy.loginfo("ready for goal location query.")
    rospy.spin()

if __name__ == "__main__":
    main()