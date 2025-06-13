export DISPLAY=:0.0
CUDA_VISIBLE_DEVICES=0,1 /home/ycs/.conda/envs/open_fusion/bin/python build_scene_graph_edges.py --data rgbd --scene wuhu_3