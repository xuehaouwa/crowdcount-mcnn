import os
import torch
import numpy as np
import cv2
from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils


# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False
vis = False
save_output = True

# data_path = './data/original/shanghaitech/part_B_final/test_data/images/'
# gt_path = './data/original/shanghaitech/part_B_final/test_data/ground_truth_csv/'
model_path = './final_models/mcnn_shtechA_660.h5'

output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()
net.training = False

im_data = cv2.imread('/home/haoxue/Pictures/Screenshot from 2018-11-05 15-26-25.png', 0)
im_data = im_data.reshape((1, 1, im_data.shape[0], im_data.shape[1]))
density_map = net(im_data)
density_map = density_map.data.cpu().numpy()
et_count = np.sum(density_map)
if vis:
    utils.display_density(im_data, density_map)
if save_output:
    utils.save_density_map(density_map, output_dir, 'output_' + '.png')
        


