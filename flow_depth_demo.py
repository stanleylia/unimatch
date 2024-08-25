import cv2
import numpy as np
import torch
import sys
import os

# Add the correct paths
sys.path.append('/content/unimatch')
sys.path.append('/content/unimatch/MiDaS')  # Add this line

from unimatch.unimatch import UniMatch
from midas.model_loader import load_model
import midas.transforms as transforms

# Rest of the code remains the same
def flow_to_image(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def estimate_depth(image, depth_model, transform, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_batch = transform({"image": image}).to(device)
    
    with torch.no_grad():
        prediction = depth_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth = prediction.cpu().numpy()
    return depth

def visualize_flow_and_depth(img, flow, depth):
    flow_img = flow_to_image(flow)
    
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_colormap = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
    
    result = cv2.addWeighted(img, 0.5, flow_img, 0.5, 0)
    result = cv2.addWeighted(result, 0.7, depth_colormap, 0.3, 0)
    
    return result

def process_video(flow_model, depth_model, transform, input_video, output_video, device, **kwargs):
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    ret, frame1 = cap.read()
    if not ret:
        return
    
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        img1 = torch.from_numpy(frame1).float().permute(2, 0, 1).unsqueeze(0).to(device)
        img2 = torch.from_numpy(frame2).float().permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            results_dict = flow_model(img1, img2, **kwargs)
        flow = results_dict['flow_preds'][-1][0].permute(1, 2, 0).cpu().numpy()
        
        depth = estimate_depth(frame1, depth_model, transform, device)
        
        vis = visualize_flow_and_depth(frame1, flow, depth)
        
        if out is None:
            out = cv2.VideoWriter(output_video, fourcc, 30, (vis.shape[1], vis.shape[0]))
        
        out.write(vis)
        frame1 = frame2
    
    cap.release()
    out.release()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load flow model
    flow_model = UniMatch(feature_channels=128, num_scales=2, upsample_factor=4, num_head=1, ffn_dim_expansion=4, num_transformer_layers=6, reg_refine=True).to(device)
    flow_model.load_state_dict(torch.load('pretrained/gmflow-scale2-regrefine6-kitti15-25b554d7.pth', map_location=device)['model'])
    flow_model.eval()
    
    # Load depth model
    model_type = "midas_v21_small"
    model_weights = "MiDaS/weights/midas_v21_small-70d6b9c8.pt"  # Update this path
    
    depth_model, transform, net_w, net_h = load_model(model_weights, model_type, device, False)
    depth_model.eval()
    
    process_video(flow_model, depth_model, transform,
                  'demo/kitti.mp4', 
                  'output/kitti/kitti_flow_depth_accurate.mp4',
                  device,
                  attn_type='swin', 
                  attn_splits_list=[2, 8], 
                  corr_radius_list=[-1, 4], 
                  prop_radius_list=[-1, 1], 
                  num_reg_refine=6)
    
    print("处理完成。输出视频保存在: output/kitti/kitti_flow_depth_accurate.mp4")

if __name__ == "__main__":
    main()
