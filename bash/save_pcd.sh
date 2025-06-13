# CUDA_VISIBLE_DEVICES=2 /home/ycs/.conda/envs/open_fusion/bin/python main.py --data icl --scene kt3 --device cuda:0
# CUDA_VISIBLE_DEVICES=0,1 /home/ycs/.conda/envs/open_fusion/bin/python main.py --data rgbd --scene wuhu_1 --device cuda:0 --save true
# CUDA_VISIBLE_DEVICES=1 /home/ycs/.conda/envs/open_fusion/bin/python main.py --data rgbd --scene wuhu_1 --device_tsdf cuda:0 --device_torch cuda:0 --save true
CUDA_VISIBLE_DEVICES=0,1 /home/ycs/.conda/envs/open_fusion/bin/python save_pcd.py --data rgbd --scene wuhu_1_inner --device_tsdf cuda:0 --device_torch cuda:0 --load true