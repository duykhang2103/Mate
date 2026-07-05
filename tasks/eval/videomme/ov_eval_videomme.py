import functools
import itertools
import logging
from tqdm import tqdm
from PIL import Image
from multiprocessing import Pool
import multiprocessing as mp
from argparse import ArgumentParser
import numpy as np

import torch
import torchvision

from decord import VideoReader, cpu
import transformers
import os

from tasks.eval.model_utils import load_llava_ov
from tasks.eval.videomme import (
    VideoMMEDataset,
    check_ans,
    save_results,
    load_results,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESOLUTION = 672


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        default='llava-hf/llava-onevision-qwen2-7b-ov-hf'
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        default='"./test_results/test_ov_videomme"'
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        default=16,
    )
    parser.add_argument(
        "--use_lora",
        action='store_true'
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        required=False,
        default=32,
    )
    parser.add_argument(
        "--weight_dir",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--conv_mode",
        type=str,
        required=False,
        default='eval_videomme',
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--selected_layer",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of splits to evaluate (e.g. 'short,medium,long'). Default: all.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of samples to evaluate per split. Default: all.",
    )
    args = parser.parse_args()
    return args


def load_model_and_dataset(rank, world_size, pretrained_model_name_or_path, num_frames, use_lora=False,
                          lora_alpha=32, weight_dir=None, selected_layer=10, alpha=0.4, tau=0.8):
    """Load LLaVA-OneVision model and VideoMME dataset."""
    model, processor = load_llava_ov(
        pretrained_model_name_or_path,
        num_frames=num_frames,
        use_lora=use_lora,
        weight_dir=weight_dir,
        lora_alpha=lora_alpha,
        selected_layer=selected_layer,
        alpha=alpha,
        tau=tau,
    )
    logger.info('Done loading LLaVA-OneVision')

    model = model.to(torch.device(rank))
    model = model.eval()

    dataset = VideoMMEDataset(num_segments=num_frames)
    dataset.set_rank_and_world_size(rank, world_size)
    return model, processor, dataset


def infer_videomme(
        args,
        model,
        processor,
        data_sample,
        pre_query_prompt=None,
        post_query_prompt=None,
        print_res=False,
    ):
    """Run inference on a single VideoMME sample using LLaVA-OV native interface."""
    video_list = data_sample["video_pils"]

    # Build question text with prompts
    question = data_sample['question']
    if pre_query_prompt:
        question = pre_query_prompt + question
    if post_query_prompt:
        question = question + post_query_prompt

    # Build conversation for LLaVA-OV using video format
    conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], videos=video_list, return_tensors="pt")
    inputs = {k: v.to(model.device) if not v.is_floating_point() else v.to(model.device, dtype=torch.bfloat16) for k, v in inputs.items()}

    with torch.no_grad():
        output_token = model.generate(**inputs,
                                      do_sample=False, max_new_tokens=args.max_new_tokens, num_beams=1,
                                      top_p=args.top_p, temperature=args.temperature, use_cache=True)
        output_text = processor.batch_decode(output_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if print_res:
        print('### PROMPTING LM WITH: ', text)
        print('### LM OUTPUT TEXT:  ', output_text)

    # Extract assistant response
    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n")[-1]
    pred = output_text.strip()

    token_info = {}
    return pred, token_info


def run(rank, args, world_size):
    """Main evaluation loop."""
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    print_res = False
    pre_query_prompt = None
    post_query_prompt = "\nOnly give the best option."

    logger.info(f'Loading model and constructing dataset to GPU {rank}...')
    model, processor, dataset = load_model_and_dataset(
        rank, world_size,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        num_frames=args.num_frames,
        use_lora=args.use_lora,
        lora_alpha=args.lora_alpha,
        weight_dir=args.weight_dir,
        selected_layer=args.selected_layer,
        alpha=args.alpha,
        tau=args.tau,
    )
    logger.info('Done loading model and dataset')

    # Filter by splits if specified
    if args.tasks:
        allowed_tasks = set(t.strip() for t in args.tasks.split(','))
        dataset.data_list = [d for d in dataset.data_list if d['task_type'] in allowed_tasks]
        logger.info(f'Filtered to splits: {allowed_tasks} ({len(dataset.data_list)} samples)')

    # Check how many videos actually exist on disk
    from collections import defaultdict
    available_counts = defaultdict(int)
    total_counts = defaultdict(int)
    for d in dataset.data_list:
        task = d['task_type']
        total_counts[task] += 1
        video_path = os.path.join(d['prefix'], d['data']['video'])
        if os.path.exists(video_path):
            available_counts[task] += 1
    for task in sorted(total_counts.keys()):
        logger.info(f'  {task}: {available_counts[task]}/{total_counts[task]} videos available on disk')

    # Limit samples per split if specified
    if args.max_samples is not None:
        task_counts = defaultdict(int)
        filtered = []
        for d in dataset.data_list:
            task = d['task_type']
            video_path = os.path.join(d['prefix'], d['data']['video'])
            if task_counts[task] < args.max_samples and os.path.exists(video_path):
                filtered.append(d)
                task_counts[task] += 1
        dataset.data_list = filtered
        logger.info(f'Limited to {args.max_samples} available samples per split ({len(dataset.data_list)} total)')

    if rank == 0:
        tbar = tqdm(total=len(dataset))

    correct = 0
    total = 0
    result_list = []
    acc_dict = {}
    done_count = 0

    for example in dataset:
        if example is None:
            continue
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0]  # correct, total
        acc_dict[task_type][1] += 1
        total += 1

        pred, token_info = infer_videomme(
            args,
            model,
            processor,
            example,
            pre_query_prompt=pre_query_prompt,
            post_query_prompt=post_query_prompt,
            print_res=print_res,
        )

        gt = example['answer']
        result_list.append({
            'pred': pred,
            'gt': gt,
            'task_type': task_type,
            'video_path': example['video_path'],
            'question': example['question'],
            'token_info': token_info,
        })

        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1

        if rank == 0:
            tbar.update(len(result_list) - done_count)
            tbar.set_description_str(
                f"Task: {task_type}, Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100:.2f}%; "
                f"Total: {correct / total * 100:.2f}%"
            )
            done_count = len(result_list)

    return result_list


def main():
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    save_path = args.save_path
    json_data = load_results(save_path)

    if json_data is None:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            logger.info(f'Started benchmarking with {n_gpus} GPUs, saving to: {save_path}')
            world_size = n_gpus
            with Pool(world_size) as pool:
                func = functools.partial(run, args=args, world_size=world_size)
                result_lists = pool.map(func, range(world_size))
            logger.info('Finished running')
            result_list = list(itertools.chain(*result_lists))
        else:
            logger.info(f'Started benchmarking (single GPU), saving to: {save_path}')
            result_list = run(0, world_size=1, args=args)
    else:
        logger.info(f'Loaded results from {save_path}')
        result_list = json_data

    save_results(result_list, save_path)


if __name__ == "__main__":
    main()
