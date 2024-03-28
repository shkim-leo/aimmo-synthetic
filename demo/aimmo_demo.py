import os
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_gif
from PIL import Image
import argparse

task_list = ["img2vid", "sim2real", "seg2real"]

def parse_args():
    parser = argparse.ArgumentParser(description="Technology Assessment Demo")
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        required=True,
        help="path to input directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        required=True,
        help="path to output directory",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="img2vid",
        choices=task_list,
        required=True,
        help="type of task to run",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="path to base model",
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        default=None,
        required=False,
        help="path to controlnet model",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        required=False,
        help="gpu device id",
    )
    args = parser.parse_args()
    return args

def image2video(args):
    #create pipeline
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(args.device)
    
    state = {'success':False, "msg":None}
    
    try:
        for image_name in os.listdir(args.input_path):
            name, ext = os.path.splitext(image_name)
            output_path = os.path.join(args.output_path, name+".gif")
            
            if ext not in [".png", ".jpg"]:
                continue
            
            image = Image.open(os.path.join(args.input_path, image_name)).convert('RGB')
            image = image.resize((1024, 576))
            
            frames = pipeline(
                image,
                num_frames=25,
                height=576,
                width=1024,
                decode_chunk_size=8,
                motion_bucket_id=100,
                min_guidance_scale=1.0,
                max_guidance_scale=3.0,
            ).frames[0]
            
            export_to_gif(frames, output_gif_path=output_path)
    except Exception as msg:
        state['msg'] = msg
        return state

    state['success'] = True
    return state

def main():
    args = parse_args()
    
    #device setting
    args.device = "cuda:{}".format(args.device)
    torch.cuda.set_device(args.device)    

    #make output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    #call function by task
    if args.task=="img2vid":
        state = image2video(args)
    elif args.task=="sim2real":
        pass
    elif args.task=="seg2real":
        pass
    else:
        raise ValueError("task must be in {}".format(task_list))

    if not state['success']:
        print(state['msg'])
    else:
        print('generation end')