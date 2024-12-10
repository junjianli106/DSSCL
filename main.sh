#!/bin/bash

python main_w_he.py --weight_orig 0.3 --num-cluster '40' --resume '/homec/wangchenghui/python_projects/Pathological_Image_Classification/SSCL/experiment_w_he/checkpoint_0020.pth.tar'
python main_w_he.py --weight_orig 0.0 --num-cluster '10'
python main_w_he.py --weight_orig 0.1 --num-cluster '10'
python main_w_he.py --weight_orig 0.2 --num-cluster '10'
python main_w_he.py --weight_orig 0.3 --num-cluster '10'