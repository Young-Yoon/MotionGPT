import torch
import time
import os
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.config import parse_args
import mGPT.render.matplot.plot_3d_global as plot_3d
from glob import glob
from llm_utils import get_response_multi_inputs
from datetime import datetime
from tqdm import tqdm

os.environ['DISPLAY'] = ':0.0'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Load model
cfg = parse_args(phase="webui")  # parse config file
cfg.FOLDER = 'cache'
output_dir = Path(cfg.FOLDER)
output_dir.mkdir(parents=True, exist_ok=True)
pl.seed_everything(cfg.SEED_VALUE)
if cfg.ACCELERATOR == "gpu":
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
datamodule = build_data(cfg, phase="test")
model = build_model(cfg, datamodule)
state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]
model.load_state_dict(state_dict)
model.to(device)


def motion_token_to_string(motion_token, lengths, codebook_size=512):
    motion_string = []
    for i in range(motion_token.shape[0]):
        motion_i = motion_token[i].cpu(
        ) if motion_token.device.type == 'cuda' else motion_token[i]
        motion_list = motion_i.tolist()[:lengths[i]]
        motion_string.append(
            (f'<motion_id_{codebook_size}>' +
             ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
             f'<motion_id_{codebook_size + 1}>'))
    return motion_string


def render_motion(data, feats, outname=None):
    fname = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(
        time.time())) + (str(np.random.randint(10000, 99999)) if not outname else '@@' + outname.replace('.npy', ''))
    video_fname = fname + '.mp4'
    feats_fname = fname + '.npy'
    output_npy_path = os.path.join(output_dir, feats_fname)
    output_mp4_path = os.path.join(output_dir, video_fname)
    np.save(output_npy_path, feats)

    output_gif_path = output_mp4_path[:-4] + '.gif'
    if len(data.shape) == 3:
        data = data[None]
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    pose_vis = plot_3d.draw_to_batch(data, [''], [output_gif_path])
    # out_video = mp.VideoFileClip(output_gif_path)
    # out_video.write_videofile(output_mp4_path)
    del pose_vis

    return output_mp4_path, video_fname, output_npy_path, feats_fname


def load_motion(motion_uploaded, file):
    filename = Path(file)

    feats = torch.tensor(np.load(file), device=model.device)
    if len(feats) == 1:
        raise ValueError("Single frame not supported")
    if len(feats.shape) == 2:
        feats = feats[None]
    feats = model.datamodule.normalize(feats)

    # Motion tokens
    motion_lengths = feats.shape[0]
    motion_token, _ = model.vae.encode(feats)

    motion_token_string = model.lm.motion_token_to_string(
        motion_token, [motion_token.shape[1]])[0]
    motion_token_length = motion_token.shape[1]

    # Motion rendered
    joints = model.datamodule.feats2joints(feats.cpu()).cpu().numpy()
    output_mp4_path, video_fname, output_npy_path, joints_fname = render_motion(
        joints,
        feats.to('cpu').numpy(), outname=filename.name)

    motion_uploaded.update({
        "feats": feats,
        "joints": joints,
        "motion_video": output_mp4_path,
        "motion_video_fname": video_fname,
        "motion_joints": output_npy_path,
        "motion_joints_fname": joints_fname,
        "motion_lengths": motion_lengths,
        "motion_token": motion_token,
        "motion_token_string": motion_token_string,
        "motion_token_length": motion_token_length,
    })

    return motion_uploaded


input_texts = [
    "Please explain the movement shown in <Motion_Placeholder> using natural language.",
    "Give me a summary of the motion being displayed in <Motion_Placeholder> using words",
    "Provide a summary of the motion demonstrated in <Motion_Placeholder> using words.",
    "Provide a summary of the motion depicted in <Motion_Placeholder> using language.",
    "Provide a description of the action in <Motion_Placeholder> using words.",
    "Describe the motion illustrated in <Motion_Placeholder> in natural language.",
    "Describe the motion represented by <Motion_Placeholder> in plain English.",
    "What kind of action is being represented in <Motion_Placeholder>? Explain it in text.",
    "Generate text for <Motion_Placeholder>:",
]


def generate_text_from_motion(motion_fname, query, sample_factor=1):
    minfo = load_motion({}, motion_fname)
    prompt = model.lm.placeholder_fulfill(query, minfo["motion_lengths"], minfo["motion_token_string"], "")
    batch = {
        "length": [minfo["motion_lengths"]],
        "text": [prompt],
    }
    outputs = model(batch, task="t2m")
    out_texts = outputs["texts"][0]
    return out_texts


split = 'train'
data_path = '/Users/ylee/data/mdm_dataset/MixamoSMPLMaleHumanML'
retry_openai = 5
# failed_list_r0.txt: 147
with open('failed_list_r0.txt', 'r') as f:
    lines = f.readlines()
failed = [int(l.split(':')[0]) for l in lines]

with open("log.txt", "a") as flog:
    flog.write(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + "\n")
    for i, fname in tqdm(enumerate(sorted(glob(os.path.join(data_path, split, "npy_files", "*.npy"))))):
        if i not in failed:
            continue
        mixamo_label = Path(fname).with_suffix('').name
        mixamo_txtfile = Path(data_path) / split / 'texts' / (mixamo_label + '.txt')
        mgpt_outfile = Path(data_path) / split / 'texts_mgpt' / (mixamo_label + '.txt')
        with open(mixamo_txtfile, 'r') as f:
            desc_mixamo = f.readlines()[0].split('#')[0]
        input_motion = f"({i}) {mixamo_label} -> {desc_mixamo}"
        print(input_motion)
        flog.write(input_motion + "\n")

        history_mgpt = []
        answer_found = False
        sample_factor = 1
        for i_mgpt, query_mgpt in enumerate(input_texts):
            try:
                mgpt_output = generate_text_from_motion(fname, query_mgpt, sample_factor=sample_factor)
            except ValueError as e:
                mgpt_output = fname

            if mgpt_output and mgpt_output not in history_mgpt:
                history_mgpt.append(mgpt_output)
            else:
                continue
            # print(f"{query_mgpt} -> {mgpt_output}")

            for i_openai in range(retry_openai):
                results = f"({i_mgpt},{i_openai}) "
                # label enhancement
                label = ', '.join([mixamo_label, desc_mixamo])
                query_openai = {"label": label, "prediction": mgpt_output}
                enhanced_label = get_response_multi_inputs("text_enhancement", query_openai)
                if enhanced_label is None:
                    print(f"{results}: {enhanced_label=}")
                    continue
                results += f"{mgpt_output} -> {enhanced_label}\n"
                query_openai["prediction"] = enhanced_label.replace('[', '').replace(']', '')
                # Reflection by chatGPT
                perf = get_response_multi_inputs("assess_prediction", query_openai)
                if perf is None:
                    print(f"{results}: {perf=}")
                    continue
                flog.write(results + perf)
                if perf.startswith("Yes") or perf.startswith("Detailed"):
                    with open(mgpt_outfile, 'w') as f:
                        f.write(query_openai["prediction"])
                    answer_found = True
                    flog.write('\n')
                    break

            if answer_found:
                break

        if not answer_found:
            with open("failed_list.txt", "a") as falied_list:
                falied_list.write(f"{i}: {mixamo_label}\n")
