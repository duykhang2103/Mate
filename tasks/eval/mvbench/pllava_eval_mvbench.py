
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

from tasks.eval.model_utils import load_pllava, pllava_answer
from tasks.eval.eval_utils import conv_templates
from tasks.eval.mvbench import (
    MVBenchDataset,
    check_ans,
    save_results,
    load_results,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESOLUTION = 672 # 


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        default='llava-hf/llava-1.5-7b-hf'
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        default='"./test_results/test_llava_mvbench"'
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        default=4,
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
        default='eval_mvbench',
    )
    parser.add_argument(
        "--pooling_shape", 
        type=str,
        required=False,
        default=None,
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
        default=0.1,
    )
    parser.add_argument(
        "--softmax", 
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--head", 
        type=int,
        default=8,
    )
    parser.add_argument(
        "--tau", 
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--temporal_segment_ratio", 
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--cluster_ratio", 
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--use_entropy_adaptive",
        action='store_true',
        help="Use entropy-adaptive dynamic layer selection instead of fixed selected_layer.",
    )
    parser.add_argument(
        "--tau_entropy",
        type=float,
        default=0.8,
        help="Entropy threshold for triggering pruning (lower = prune later).",
    )
    parser.add_argument(
        "--entropy_fallback_layer",
        type=int,
        default=20,
        help="Fallback layer if no entropy trigger found.",
    )
    parser.add_argument(
        "--use_motion_adaptive",
        action='store_true',
        help="Use motion-adaptive dynamic alpha per window instead of fixed alpha.",
    )
    parser.add_argument(
        "--motion_scale",
        type=float,
        default=0.5,
        help="Scale factor for motion-adaptive alpha (alpha = motion_scale * motion_score).",
    )
    parser.add_argument(
        "--motion_invert",
        action='store_true',
        help="Invert motion-adaptive: high motion -> FEWER tokens (motion = noise).",
    )
    parser.add_argument(
        "--use_borderline_preservation",
        action='store_true',
        help="Use borderline token preservation (keep tokens near the pruning cutoff).",
    )
    parser.add_argument(
        "--borderline_margin",
        type=float,
        default=0.1,
        help="Margin as fraction of cutoff score (0.1 = keep tokens within 10% of cutoff).",
    )
    parser.add_argument(
        "--use_weighted_merge",
        action='store_true',
        default=True,
        help="Use similarity-weighted token merge (default: True).",
    )
    parser.add_argument(
        "--no_weighted_merge",
        action='store_true',
        default=False,
        help="Disable weighted merge, use 50/50 average instead.",
    )
    parser.add_argument(
        "--use_flow_pruning",
        action='store_true',
        default=False,
        help="Use optical flow magnitude for static/dynamic token classification instead of feature similarity.",
    )
    parser.add_argument(
        "--flow_dynamic_ratio",
        type=float,
        default=0.5,
        help="Fraction of tokens classified as dynamic based on flow magnitude (0.5 = top 50%% are dynamic).",
    )
    parser.add_argument(
        "--tasks", 
        type=str,
        default=None,
        help="Comma-separated list of task types to evaluate on (e.g. 'Action Sequence,State Change'). Default: all tasks.",
    )
    parser.add_argument(
        "--max_samples", 
        type=int,
        default=None,
        help="Max number of samples to evaluate per task. Default: all samples.",
    )
    args = parser.parse_args()
    return args

def load_model_and_dataset(rank, world_size, pretrained_model_name_or_path, num_frames, use_lora, lora_alpha, weight_dir, pooling_shape=(16,12,12), selected_layer=10, alpha=0.1, softmax=1.0, head=0, tau=1.0, cluster_ratio=1.0, temporal_segment_ratio=1.0, use_entropy_adaptive=False, tau_entropy=0.8, entropy_fallback_layer=20, use_motion_adaptive=False, motion_scale=0.5, motion_invert=False, use_borderline_preservation=False, borderline_margin=0.1, use_weighted_merge=True, use_flow_pruning=False, flow_dynamic_ratio=0.5):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    model, processor = load_pllava(pretrained_model_name_or_path, num_frames=num_frames, use_lora=use_lora, \
        weight_dir=weight_dir, lora_alpha=lora_alpha, pooling_shape=pooling_shape, selected_layer=selected_layer, \
            alpha=alpha, softmax=softmax, head=head, tau=tau, cluster_ratio=cluster_ratio, temporal_segment_ratio=temporal_segment_ratio, \
                use_entropy_adaptive=use_entropy_adaptive, tau_entropy=tau_entropy, entropy_fallback_layer=entropy_fallback_layer,
                use_motion_adaptive=use_motion_adaptive, motion_scale=motion_scale, motion_invert=motion_invert,
                use_borderline_preservation=use_borderline_preservation, borderline_margin=borderline_margin,
                use_weighted_merge=use_weighted_merge,
                use_flow_pruning=use_flow_pruning, flow_dynamic_ratio=flow_dynamic_ratio)
    logger.info('done loading llava')

    #  position embedding
    model = model.to(torch.device(rank))
    model = model.eval()

    dataset = MVBenchDataset(num_segments=num_frames)
    dataset.set_rank_and_world_size(rank, world_size)
    return model, processor, dataset

def infer_mvbench(
        args,
        model,
        processor,
        data_sample,
        conv_mode,
        pre_query_prompt=None, # add in the head of question
        post_query_prompt=None, # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt=None,  # add in the begining of return message
        print_res=False,
    ):
    video_list = data_sample["video_pils"]
    conv = conv_templates[conv_mode].copy()
    conv.user_query(data_sample['question'], pre_query_prompt, post_query_prompt, is_mm=True)
    if answer_prompt is not None:
        conv.assistant_response(answer_prompt)
    
    llm_message, conv, token_info = pllava_answer(
        conv=conv,
        model=model,
        processor=processor,
        img_list=video_list,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        print_res=print_res,
        top_p=args.top_p,
        temperature=args.temperature
    )
    
    if answer_prompt is not None:
        llm_message =  ''.join(llm_message.split(answer_prompt)[1:])

    if return_prompt is not None:
        llm_message = return_prompt + llm_message

    return llm_message, token_info
    
def single_test(args, model, processor, vid_path, num_frames=4, conv_mode="plain"):
    def get_index(num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(video_path, num_segments=8, return_msg=False, num_frames=4, resolution=336):
        transforms = torchvision.transforms.Resize(size=resolution)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        num_frames = len(vr)
        frame_indices = get_index(num_frames, num_segments)
        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(transforms(img))
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return images_group, msg
        else:
            return images_group

    if num_frames != 0:
        vid, msg = load_video(vid_path, num_segments=num_frames, return_msg=True, resolution=RESOLUTION)
    else:
        vid, msg = None, 'num_frames is 0, not inputing image'
    img_list = vid
    conv = conv_templates[conv_mode].copy()
    conv.user_query("Describe the video in details.", is_mm=True)
    llm_response, conv, _ = pllava_answer(conv=conv, model=model, processor=processor, do_sample=False, img_list=img_list, max_new_tokens=args.max_new_tokens, print_res=True)

def run(rank, args, world_size):
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    print_res = False
    conv_mode= args.conv_mode
    pre_query_prompt = None
    post_query_prompt = "\nOnly give the best option."
    if args.pooling_shape is not None:
        pooling_shape=tuple([int(x) for x in args.pooling_shape.split("-")])
    else:
        pooling_shape=(16,12,12)

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    model, processor, dataset = load_model_and_dataset(rank,
                                                       world_size,
                                                       pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                                       num_frames=args.num_frames,
                                                       use_lora=args.use_lora,
                                                       lora_alpha=args.lora_alpha,
                                                       weight_dir=args.weight_dir,
                                                       pooling_shape=pooling_shape,
                                                       selected_layer=args.selected_layer,
                                                       alpha=args.alpha,
                                                       softmax=args.softmax,
                                                       head=args.head,
                                                       tau=args.tau,
                                                       temporal_segment_ratio=args.temporal_segment_ratio,
                                                       cluster_ratio=args.cluster_ratio,
                                                       use_entropy_adaptive=args.use_entropy_adaptive,
                                                       tau_entropy=args.tau_entropy,
                                                        entropy_fallback_layer=args.entropy_fallback_layer,
                                                          use_motion_adaptive=args.use_motion_adaptive,
                                                          motion_scale=args.motion_scale,
                                                          motion_invert=args.motion_invert,
                                                           use_borderline_preservation=args.use_borderline_preservation,
                                                           borderline_margin=args.borderline_margin,
                                                           use_weighted_merge=args.use_weighted_merge and not args.no_weighted_merge,
                                                           use_flow_pruning=args.use_flow_pruning,
                                                           flow_dynamic_ratio=args.flow_dynamic_ratio)
    logger.info(f'done model and dataset...')
    logger.info('constructing dataset...')

    # Filter by tasks if specified
    if args.tasks:
        allowed_tasks = set(t.strip() for t in args.tasks.split(','))
        dataset.data_list = [d for d in dataset.data_list if d['task_type'] in allowed_tasks]
        logger.info(f'Filtered to tasks: {allowed_tasks} ({len(dataset.data_list)} samples)')

    # Limit samples per task if specified
    if args.max_samples is not None:
        from collections import defaultdict
        task_counts = defaultdict(int)
        filtered = []
        for d in dataset.data_list:
            task = d['task_type']
            if task_counts[task] < args.max_samples:
                filtered.append(d)
                task_counts[task] += 1
        dataset.data_list = filtered
        logger.info(f'Limited to {args.max_samples} samples per task ({len(dataset.data_list)} total)')

    logger.info('single test...')

    vid_path = "./example/yoga.mp4"
    # vid_path = "./example/jesse_dance.mp4"
    if rank == 0:
        single_test(args, model,
                    processor,
                    vid_path,
                    num_frames=args.num_frames,
                    conv_mode=args.conv_mode)
        logger.info('single test done...')
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
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        pred, token_info = infer_mvbench(
            args,
            model,
            processor,
            example,
            conv_mode=conv_mode,
            pre_query_prompt=pre_query_prompt,
            post_query_prompt=post_query_prompt,
            answer_prompt="Best option:(",
            return_prompt='(',
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
            tbar.update(len(result_list) - done_count, )
            tbar.set_description_str(
                f"One Chunk--Task Type: {task_type}, Chunk Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%;" 
                f" Chunk Total Acc: {correct / total * 100 :.2f}%"
            )
            done_count = len(result_list)
    return result_list

def main():
    args = parse_args()
    save_path = args.save_path
    json_data = load_results(save_path)
    if json_data is None:
        logger.info(f'started benchmarking, saving to: {save_path}')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        if world_size > 1:
            mp.set_start_method('spawn', force=True)
            with Pool(world_size) as pool:
                func = functools.partial(run, args=args, world_size=world_size)
                result_lists = pool.map(func, range(world_size))
            result_list = list(itertools.chain(*result_lists))
        else:
            result_list = run(0, args=args, world_size=1)
        logger.info('finished running')
    else:
        logger.info(f'loaded results from {save_path}')
        result_list = json_data
    save_results(result_list, save_path)
    
    
if __name__ == "__main__":
    main()