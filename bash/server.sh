source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export PYTHONPATH=$PYTHONPATH:~/catkin_ws/src/goal_location_query/
export ROS_MASTER_URI=http://172.17.129.146:11311/
export ROS_IP=192.168.10.100
CUDA_VISIBLE_DEVICES=1 /home/ycs/.conda/envs/open_fusion/bin/python ros_query.py --data rgbd --scene wuhu_1 --device_tsdf cuda:0 --device_torch cuda:0 --load true