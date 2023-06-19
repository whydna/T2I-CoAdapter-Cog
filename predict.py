from cog import BasePredictor, File, Path, Input
import tempfile
import os
from typing import List
from PIL import Image
import numpy

# demo inspired by https://huggingface.co/spaces/lambdalabs/image-mixer-demo
import argparse
import copy
import cv2
import torch
from itertools import chain
from torch import autocast
from pytorch_lightning import seed_everything

from basicsr.utils import tensor2img
from ldm.inference_base import DEFAULT_NEGATIVE_PROMPT, diffusion_inference, get_adapters, get_sd_models
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import ExtraCondition, get_cond_model
from ldm.modules.encoders.adapter import CoAdapterFuser

torch.set_grad_enabled(False)

supported_cond = ['style', 'sketch', 'color', 'depth', 'canny']

# config
parser = argparse.ArgumentParser()
parser.add_argument(
    '--sd_ckpt',
    type=str,
    default='models/v1-5-pruned-emaonly.ckpt',
    help='path to checkpoint of stable diffusion model, both .ckpt and .safetensor are supported',
)
parser.add_argument(
    '--vae_ckpt',
    type=str,
    default=None,
    help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded',
)
global_opt = parser.parse_args()
global_opt.config = 'configs/stable-diffusion/sd-v1-inference.yaml'
for cond_name in supported_cond:
    setattr(global_opt, f'{cond_name}_adapter_ckpt', f'models/coadapter-{cond_name}-sd15v1.pth')
global_opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
global_opt.max_resolution = 512 * 512
global_opt.sampler = 'ddim'
global_opt.cond_weight = 1.0
global_opt.C = 4
global_opt.f = 8
#TODO: expose style_cond_tau to users
global_opt.style_cond_tau = 1.0

# stable-diffusion model
sd_model, sampler = get_sd_models(global_opt)
# adapters and models to processing condition inputs
adapters = {}
cond_models = {}

torch.cuda.empty_cache()

# fuser is indispensable
coadapter_fuser = CoAdapterFuser(unet_channels=[320, 640, 1280, 1280], width=768, num_head=8, n_layes=3)
coadapter_fuser.load_state_dict(torch.load(f'models/coadapter-fuser-sd15v1.pth'))
coadapter_fuser = coadapter_fuser.to(global_opt.device)

class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(self,
        prompt: str = Input(),
        neg_prompt: str = Input(description="Negative prompt", default=""),
        depth_img: File = Input(default=None),
        depth_weight: float = Input(default=1.0),
        canny_img: File = Input(default=None),
        canny_weight: float = Input(default=1.0),
        sketch_img: File = Input(default=None),
        sketch_weight: float = Input(default=1.0),
        color_img: File = Input(default=None),
        color_weight: float = Input(default=1.0),
        style_img: File = Input(default=None),
        style_weight: float = Input(default=1.0),
        n_samples: int = Input(description="Number of samples to generate", default=1),
        resize_short_edge: int = Input(description="Image resolution", default=512),
        scale: float = Input(description="Guidance Scale (CFG)", default=7.5),
        steps: float = Input(description="Steps", default=50),
        seed: int = Input(description="Seed", default=42),
        cond_tau : float = Input(
            description="timestamp parameter that determines until which step the adapter is applied",
            default=1.0
        )
    ) -> List[Path]:
        # ['style', 'sketch', 'color', 'depth', 'canny']
        btns = [
            "Image" if style_img else "Nothing",
            "Image" if sketch_img else "Nothing",
            "Image" if color_img else "Nothing",
            "Image" if depth_img else "Nothing",
            "Image" if canny_img else "Nothing",
        ]
        # source image - order matters (see supported_conds)
        ims1 = [
            numpy.array(Image.open(style_img).convert('RGB')) if style_img else None,
            numpy.array(Image.open(sketch_img).convert('RGB')) if sketch_img else None,
            numpy.array(Image.open(color_img).convert('RGB')) if color_img else None,
            numpy.array(Image.open(depth_img).convert('RGB')) if depth_img else None,
            numpy.array(Image.open(canny_img).convert('RGB')) if canny_img else None,
        ]
        # conditioned images
        ims2 = [None,None,None,None,None]
        cond_weights = [style_weight,sketch_weight,color_weight,depth_weight,canny_weight]
        inps = list(chain(btns, ims1, ims2, cond_weights))
        inps.extend([prompt, neg_prompt, scale, n_samples, seed, steps, resize_short_edge, cond_tau])

        [outputs, conds] = self.run(*inps)

        print(f"Returned {len(outputs)} outputs and {len(conds)} conds")
        
        output_dir = Path(tempfile.mkdtemp())

        output_paths = []

        for i, output in enumerate(outputs):
            path = os.path.join(output_dir, f'{i:05}_output.png') 
            cv2.imwrite(path, output)
            output_paths.append(Path(path))

        for i, cond in enumerate(conds):
            path = os.path.join(output_dir, f'{i:05}_${cond}.png') 
            cv2.imwrite(path, cond)
            output_paths.append(Path(path))

        print(output_paths)

        return output_paths

    def run(self, *args):
        print(args)
        with torch.inference_mode(), \
                sd_model.ema_scope(), \
                autocast('cuda'):

            inps = []
            for i in range(0, len(args) - 8, len(supported_cond)):
                inps.append(args[i:i + len(supported_cond)])

            print(inps)

            opt = copy.deepcopy(global_opt)
            opt.prompt, opt.neg_prompt, opt.scale, opt.n_samples, opt.seed, opt.steps, opt.resize_short_edge, opt.cond_tau \
                = args[-8:]

            conds = []
            activated_conds = []
            prev_size = None
            for idx, (b, im1, im2, cond_weight) in enumerate(zip(*inps)):
                cond_name = supported_cond[idx]
                if b == 'Nothing':
                    if cond_name in adapters:
                        adapters[cond_name]['model'] = adapters[cond_name]['model'].cpu()
                else:
                    activated_conds.append(cond_name)
                    if cond_name in adapters:
                        adapters[cond_name]['model'] = adapters[cond_name]['model'].to(opt.device)
                    else:
                        adapters[cond_name] = get_adapters(opt, getattr(ExtraCondition, cond_name))
                    adapters[cond_name]['cond_weight'] = cond_weight

                    process_cond_module = getattr(api, f'get_cond_{cond_name}')

                    if b == 'Image':
                        if cond_name not in cond_models:
                            cond_models[cond_name] = get_cond_model(opt, getattr(ExtraCondition, cond_name))
                        if prev_size is not None:
                            image = cv2.resize(im1, prev_size, interpolation=cv2.INTER_LANCZOS4)
                        else:
                            image = im1
                        conds.append(process_cond_module(opt, image, 'image', cond_models[cond_name]))
                        if idx != 0 and prev_size is None:  # skip style since we only care spatial cond size
                            h, w = image.shape[:2]
                            prev_size = (w, h)
                    else:
                        if prev_size is not None:
                            image = cv2.resize(im2, prev_size, interpolation=cv2.INTER_LANCZOS4)
                        else:
                            image = im2
                        conds.append(process_cond_module(opt, image, cond_name, None))
                        if idx != 0 and prev_size is None:  # skip style since we only care spatial cond size
                            h, w = image.shape[:2]
                            prev_size = (w, h)



            features = dict()
            for idx, cond_name in enumerate(activated_conds):
                cur_feats = adapters[cond_name]['model'](conds[idx])
                if isinstance(cur_feats, list):
                    for i in range(len(cur_feats)):
                        cur_feats[i] *= adapters[cond_name]['cond_weight']
                else:
                    cur_feats *= adapters[cond_name]['cond_weight']
                features[cond_name] = cur_feats

            adapter_features, append_to_context = coadapter_fuser(features)

            output_conds = []
            for cond in conds:
                output_conds.append(tensor2img(cond, rgb2bgr=False))

            ims = []
            seed_everything(opt.seed)
            for _ in range(opt.n_samples):
                result = diffusion_inference(opt, sd_model, sampler, adapter_features, append_to_context)
                ims.append(tensor2img(result, rgb2bgr=False))

            # Clear GPU memory cache so less likely to OOM
            torch.cuda.empty_cache()
            return ims, output_conds

