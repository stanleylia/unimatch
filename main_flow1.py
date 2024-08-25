import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import argparse
import numpy as np
import os
import cv2

from dataloader.flow.datasets import build_train_dataset
from unimatch.unimatch import UniMatch
from loss.flow_loss import flow_loss_func

from evaluate_flow import (validate_chairs, validate_things, validate_sintel, validate_kitti,
                           create_kitti_submission, create_sintel_submission,
                           inference_flow,
                           )

from utils.logger import Logger
from utils import misc
from utils.dist_utils import get_dist_info, init_dist, setup_for_distributed

# ... [rest of the imports and functions remain the same] ...

def inference_flow(model,
                   inference_dir=None,
                   inference_video=None,
                   output_path='output',
                   padding_factor=8,
                   inference_size=None,
                   save_flo_flow=False,
                   attn_type='swin',
                   attn_splits_list=None,
                   corr_radius_list=None,
                   prop_radius_list=None,
                   num_reg_refine=1,
                   pred_bidir_flow=False,
                   pred_bwd_flow=False,
                   fwd_bwd_consistency_check=False,
                   save_video=False,
                   concat_flow_img=False,
                   ):
    # ... [rest of the function remains the same until the video processing part] ...

    if inference_video is not None:
        video = cv2.VideoCapture(inference_video)
        fps = video.get(cv2.CAP_PROP_FPS)

        if save_video:
            # Get the original video's width and height
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if concat_flow_img:
                out_width = width * 2
                out_height = height
            else:
                out_width = width
                out_height = height

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(os.path.join(output_path, 'output_flow.mp4'), fourcc, fps, (out_width, out_height))

            if pred_bidir_flow:
                out_bwd = cv2.VideoWriter(os.path.join(output_path, 'output_flow_bwd.mp4'), fourcc, fps,
                                          (out_width, out_height))

        ret, frame1 = video.read()
        if ret:
            pass
        else:
            print('Empty video')
            return

        while True:
            ret, frame2 = video.read()

            if not ret:
                break

            img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            img1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0)
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0)

            img1, img2 = img1.cuda(), img2.cuda()

            results_dict = model(img1, img2,
                                 attn_type=attn_type,
                                 attn_splits_list=attn_splits_list,
                                 corr_radius_list=corr_radius_list,
                                 prop_radius_list=prop_radius_list,
                                 num_reg_refine=num_reg_refine,
                                 )

            flow = results_dict['flow_preds'][-1]

            if pred_bidir_flow:
                flow_bwd = results_dict['flow_preds_bwd'][-1]

            img1 = img1[0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            img2 = img2[0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)

            flow = flow[0].cpu().permute(1, 2, 0).numpy()

            if pred_bidir_flow:
                flow_bwd = flow_bwd[0].cpu().permute(1, 2, 0).numpy()

            # computing flow magnitude for pseudo-depth
            flow_magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
            max_magnitude = np.max(flow_magnitude)
            normalized_depth = (flow_magnitude / max_magnitude * 255).astype(np.uint8)
            
            # create a color map
            depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)

            # add text to show depth range
            cv2.putText(depth_colormap, "0 cm", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(depth_colormap, f"{int(max_magnitude)} cm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if save_video:
                if concat_flow_img:
                    output_img = np.concatenate((img2, depth_colormap), axis=1)
                else:
                    output_img = depth_colormap

                out.write(output_img)

            if pred_bidir_flow:
                flow_bwd = flow_bwd_to_rgb(flow_bwd)

                if save_video:
                    if concat_flow_img:
                        output_img_bwd = np.concatenate((img2, flow_bwd), axis=1)
                    else:
                        output_img_bwd = flow_bwd

                    out_bwd.write(output_img_bwd)

            frame1 = frame2

        video.release()

        if save_video:
            out.release()
            if pred_bidir_flow:
                out_bwd.release()

    print(f'Visualization results have been saved to {output_path}')

# ... [rest of the code remains the same] ...

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    main(args)
