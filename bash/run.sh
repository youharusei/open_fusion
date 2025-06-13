export DISPLAY=:0.0
CUDA_VISIBLE_DEVICES=0,1 /home/ycs/.conda/envs/open_fusion/bin/python main.py --data rgbd --scene wuhu_1_inner --device_tsdf cuda:1 --device_torch cuda:0 --save true