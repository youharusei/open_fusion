export DISPLAY=:0.0
CUDA_VISIBLE_DEVICES=0,1 /home/ycs/.conda/envs/open_fusion/bin/python build_scene_graph.py --data rgbd --scene wuhu_1 --device_tsdf cuda:1 --device_torch cuda:0