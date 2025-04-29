import os
import cv2
import torch
import argparse
from doclayout_yolo import YOLOv10

import pdb

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, required=True, type=str)
    parser.add_argument('--image-path', default=None, required=True, type=str)
    parser.add_argument('--res-path', default='outputs', required=False, type=str)
    parser.add_argument('--imgsz', default=1024, required=False, type=int)
    parser.add_argument('--line-width', default=5, required=False, type=int)
    parser.add_argument('--font-size', default=20, required=False, type=int)
    parser.add_argument('--conf', default=0.2, required=False, type=float)
    args = parser.parse_args()
    
    # Automatically select device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = YOLOv10(args.model)  # load an official model

    det_res = model.predict(
        args.image_path,
        imgsz=args.imgsz,
        conf=args.conf,
        device=device,
    )

    # # import pdb; pdb.set_trace()

    summaries = det_res[0].summary()

    ### bounding box crop
    print(summaries[0])

    for idx, summary in enumerate(summaries) : 
        if summary['name'] == 'plain text' : 
            output_dir = os.path.join(args.res_path, summary['name'])
            os.makedirs(output_dir, exist_ok=True)
            x1, y1, x2, y2 = summary['box']['x1'], summary['box']['y1'], summary['box']['x2'], summary['box']['y2']
            crop_img = det_res[0].orig_img[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite(os.path.join(output_dir, f"{args.image_path.split('/')[-1].replace('.jpg', '')}_{idx}.jpg"), crop_img)

    annotated_frame = det_res[0].plot(pil=True, line_width=args.line_width, font_size=args.font_size)

    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)
    output_path = os.path.join(args.res_path, args.image_path.split("/")[-1].replace(".jpg", "_res.jpg"))
    cv2.imwrite(output_path, annotated_frame)
    print(f"Result saved to {output_path}")