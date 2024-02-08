import torch
from PIL import Image
from diffusers.utils import load_image
from diffusers import StableDiffusionXLReferencePipeline
from diffusers.schedulers import UniPCMultistepScheduler

input_image = load_image("/data/noah/inference/reimagine_dataset/images/1652248040218_RR-View-CMR-Wide.png")

pipe = StableDiffusionXLReferencePipeline.from_pretrained(
    "/data/noah/ckpt/pretrain_ckpt/StableDiffusion/sdxl",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda:3")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

result_img = pipe(
    ref_image=input_image,
    prompt="cars are driving",
    num_inference_steps=20,
    reference_attn=True,
    reference_adain=True,
    original_size=(1024, 1024),
    target_size=(1024, 1024),
).images[0]

result_img
