
import functools
import itertools
import json
import logging
import time
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
from tasks.eval.videomme import (
    VideoMMEDataset,
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
        default='eval_videomme',
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
        "--head", 
        type=int,
        default=0,
    )
    parser.add_argument(
        "--softmax", 
        type=float,
        default=1.0,
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
        help="Scale factor for motion-adaptive alpha (alpha = alpha * (1 + motion_scale)).",
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
        "--use_query_guided_merge",
        action='store_true',
        default=False,
        help="Use question relevance plus semantic novelty for early static/dynamic token routing.",
    )
    parser.add_argument(
        "--query_merge_weight",
        type=float,
        default=0.7,
        help="Maximum weight assigned to question relevance; flat query scores are down-weighted automatically.",
    )
    parser.add_argument(
        "--query_dynamic_ratio",
        type=float,
        default=0.5,
        help="Fixed fraction of spatial locations preserved as per-frame dynamic tokens.",
    )
    parser.add_argument(
        "--use_distortion_routing",
        action='store_true',
        default=False,
        help="Use SafePruneVid global distortion-per-token routing.",
    )
    parser.add_argument(
        "--distortion_epsilon",
        type=float,
        default=0.02,
        help="Global minimum distortion reduction per extra dynamic token.",
    )
    parser.add_argument(
        "--distortion_use_query_relevance",
        action='store_true',
        default=False,
        help="Multiply distortion value by validated question relevance.",
    )
    parser.add_argument(
        "--distortion_query_weight",
        type=float,
        default=0.7,
        help="Question relevance multiplier for distortion routing.",
    )
    parser.add_argument(
        "--disable_efficiency_metrics",
        action='store_true',
        default=False,
        help="Disable synchronized compressor and prefill timing.",
    )
    parser.add_argument(
        "--safety_router_path",
        type=str,
        default=None,
        help="Optional trained logistic safety-router JSON; requires distortion routing.",
    )
    parser.add_argument(
        "--use_cluster_pruning",
        action='store_true',
        default=False,
        help="Use DPC-KNN clustering on LLM features instead of attention-based top-k selection.",
    )
    parser.add_argument(
        "--cluster_pruning_topk",
        type=float,
        default=0.4,
        help="Fraction of clusters to keep when use_cluster_pruning is enabled.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated list of splits to evaluate (e.g. 'short,medium'). Default: all.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of samples to evaluate per split. Default: all.",
    )
    parser.add_argument(
        "--max_videos_per_split",
        type=int,
        default=None,
        help="Retain all questions for at most this many unique videos per duration split.",
    )
    args = parser.parse_args()
    return args

def load_model_and_dataset(rank, world_size, pretrained_model_name_or_path, num_frames, use_lora, lora_alpha, weight_dir, pooling_shape=(16,12,12), selected_layer=10, alpha=0.1, softmax=1.0, head=0, tau=1.0, cluster_ratio=1.0, temporal_segment_ratio=1.0, use_entropy_adaptive=False, tau_entropy=0.8, entropy_fallback_layer=20, use_motion_adaptive=False, motion_scale=0.5, motion_invert=False, use_borderline_preservation=False, borderline_margin=0.1, use_weighted_merge=True, use_flow_pruning=False, flow_dynamic_ratio=0.5, use_cluster_pruning=False, cluster_pruning_topk=0.4):
    # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
    model, processor = load_pllava(pretrained_model_name_or_path, num_frames=num_frames, use_lora=use_lora, \
        weight_dir=weight_dir, lora_alpha=lora_alpha, pooling_shape=pooling_shape, selected_layer=selected_layer, \
            alpha=alpha, softmax=softmax, head=head, tau=tau, cluster_ratio=cluster_ratio, temporal_segment_ratio=temporal_segment_ratio, \
                use_entropy_adaptive=use_entropy_adaptive, tau_entropy=tau_entropy, entropy_fallback_layer=entropy_fallback_layer,
                use_motion_adaptive=use_motion_adaptive, motion_scale=motion_scale, motion_invert=motion_invert,
                use_borderline_preservation=use_borderline_preservation, borderline_margin=borderline_margin,
                use_weighted_merge=use_weighted_merge,
                use_flow_pruning=use_flow_pruning, flow_dynamic_ratio=flow_dynamic_ratio,
                use_cluster_pruning=use_cluster_pruning, cluster_pruning_topk=cluster_pruning_topk)
    logger.info('done loading llava')

    #  position embedding
    model = model.to(torch.device(rank))
    model = model.eval()

    dataset = VideoMMEDataset(num_segments=num_frames)
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

    model_device = getattr(model, 'device', None)
    measure_cuda = (
        torch.cuda.is_available()
        and model_device is not None
        and str(model_device).startswith('cuda')
    )
    if measure_cuda:
        torch.cuda.synchronize(model_device)
        torch.cuda.reset_peak_memory_stats(model_device)
    inference_started = time.perf_counter()

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
    if measure_cuda:
        torch.cuda.synchronize(model_device)
    token_info['total_latency_ms'] = (
        time.perf_counter() - inference_started
    ) * 1000.0
    token_info['peak_gpu_memory_mb'] = (
        torch.cuda.max_memory_allocated(model_device) / (1024.0 ** 2)
        if measure_cuda
        else None
    )
    query_reliability = getattr(
        model, '_last_query_merge_reliability', None
    )
    if query_reliability is not None:
        token_info['query_merge_reliability'] = getattr(
            model, '_last_query_merge_reliability', None
        )
    token_info['prefill_tokens'] = getattr(
        model, '_last_prefill_token_count', None
    )
    token_info['prefill_latency_ms'] = getattr(
        model, '_last_prefill_latency_ms', None
    )
    token_info['compressor_latency_ms'] = getattr(
        model, '_last_compressor_latency_ms', None
    )
    routing_diagnostics = getattr(
        model, '_last_distortion_routing', None
    )
    if routing_diagnostics:
        token_info.update({
            f'routing_{name}': value
            for name, value in routing_diagnostics.items()
        })
        token_info['routing_window_sizes'] = getattr(
            model, '_last_distortion_window_sizes', None
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
    if args.use_distortion_routing and (
        args.use_query_guided_merge or args.use_flow_pruning
    ):
        raise ValueError(
            "distortion routing cannot be combined with fixed query or flow routing"
        )
    if (
        args.distortion_use_query_relevance
        and not args.use_distortion_routing
    ):
        raise ValueError(
            "--distortion_use_query_relevance requires --use_distortion_routing"
        )
    if args.distortion_epsilon < 0:
        raise ValueError("--distortion_epsilon must be non-negative")
    if args.safety_router_path and not args.use_distortion_routing:
        raise ValueError(
            "--safety_router_path requires --use_distortion_routing"
        )
    if args.max_samples is not None and args.max_videos_per_split is not None:
        raise ValueError(
            "use only one of --max_samples and --max_videos_per_split"
        )
    if args.max_videos_per_split is not None and args.max_videos_per_split <= 0:
        raise ValueError("--max_videos_per_split must be positive")

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
        pooling_shape=(16, 12, 12)

    weight_dir = args.weight_dir or args.pretrained_model_name_or_path

    if rank == 0:
        logger.info("Dataset path preflight:")
        logger.info("  cwd: %s", os.getcwd())
        logger.info("  dataset json root: %s (exists=%s)", os.path.abspath(VideoMMEDataset.data_dir), os.path.exists(VideoMMEDataset.data_dir))
        for split_name, (json_name, data_root, _, _) in VideoMMEDataset.data_list_info.items():
            json_path = os.path.join(VideoMMEDataset.data_dir, json_name)
            logger.info("  split=%s", split_name)
            logger.info("    json_path: %s (exists=%s)", os.path.abspath(json_path), os.path.exists(json_path))
            logger.info("    data_root: %s (exists=%s)", os.path.abspath(data_root), os.path.exists(data_root))

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    model, processor, dataset = load_model_and_dataset(rank,
                                                       world_size,
                                                       pretrained_model_name_or_path=args.pretrained_model_name_or_path,
                                                       num_frames=args.num_frames,
                                                       use_lora=args.use_lora,
                                                       lora_alpha=args.lora_alpha,
                                                       weight_dir=weight_dir,
                                                       pooling_shape=pooling_shape,
                                                       selected_layer=args.selected_layer,
                                                       alpha=args.alpha,
                                                       head=args.head,
                                                       softmax=args.softmax,
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
                                                         flow_dynamic_ratio=args.flow_dynamic_ratio,
                                                         use_cluster_pruning=args.use_cluster_pruning,
                                                         cluster_pruning_topk=args.cluster_pruning_topk)
    model.config.use_query_guided_merge = args.use_query_guided_merge
    model.config.query_merge_weight = args.query_merge_weight
    model.config.query_dynamic_ratio = args.query_dynamic_ratio
    model.config.use_distortion_routing = args.use_distortion_routing
    model.config.distortion_epsilon = args.distortion_epsilon
    model.config.distortion_use_query_relevance = (
        args.distortion_use_query_relevance
    )
    model.config.distortion_query_weight = args.distortion_query_weight
    model.config.record_efficiency_metrics = (
        not args.disable_efficiency_metrics
    )
    model.config.safety_router = None
    if args.safety_router_path:
        with open(args.safety_router_path, "r", encoding="utf-8") as handle:
            safety_router = json.load(handle)
        if safety_router.get("type") != "logistic_safety_router":
            raise ValueError("unsupported safety-router artifact type")
        model.config.safety_router = safety_router
    logger.info(f'done model and dataset...')
    logger.info('constructing dataset...')

    # Diagnostic logs to trace why no results are produced
    logger.info('Dataset diagnostics:')
    try:
        logger.info('  reported dataset length (len(dataset)): %s', len(dataset))
    except Exception:
        logger.exception('  failed to get len(dataset)')
    # show a few example entries from data_list (no large dumps)
    for i, d in enumerate(dataset.data_list[:5]):
        sample_video = os.path.join(d['prefix'], d['data'].get('video', '<no-video>'))
        logger.info('  sample[%d] task=%s video=%s exists=%s', i, d['task_type'], sample_video, os.path.exists(sample_video))

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

    if args.max_videos_per_split is not None:
        selected_videos = defaultdict(set)
        filtered = []
        for d in dataset.data_list:
            task = d['task_type']
            video_path = os.path.join(d['prefix'], d['data']['video'])
            if not os.path.exists(video_path):
                continue
            if (
                video_path in selected_videos[task]
                or len(selected_videos[task]) < args.max_videos_per_split
            ):
                selected_videos[task].add(video_path)
                filtered.append(d)
        dataset.data_list = filtered
        selected_summary = {
            task: len(paths) for task, paths in selected_videos.items()
        }
        logger.info(
            'Limited unique videos per split to %s: %s (%s questions)',
            args.max_videos_per_split,
            selected_summary,
            len(dataset.data_list),
        )

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
    skipped_examples = 0
    inference_errors = 0

    for example in dataset:
        if example is None:
            skipped_examples += 1
            logger.debug('Skipped a dataset entry (returned None) -- skipped so far: %s', skipped_examples)
            continue
        task_type = example['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1

        try:
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
        except Exception:
            inference_errors += 1
            logger.exception('Inference failed for video %s', example.get('video_path'))
            continue
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
    mp.set_start_method('spawn', force=True)
    args = parse_args()
    save_path = args.save_path
    json_data = load_results(save_path)
    if json_data is None:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            logger.info(f'started benchmarking with {n_gpus} GPUs, saving to: {save_path}')
            world_size = n_gpus
            with Pool(world_size) as pool:
                func = functools.partial(run, args=args, world_size=world_size)
                result_lists = pool.map(func, range(world_size))
            logger.info('finished running')
            result_list = [ res for res in itertools.chain(*result_lists)]
        else:
            logger.info(f'started benchmarking (single GPU), saving to: {save_path}')
            result_list = run(0, world_size=1, args=args)

    else:
        logger.info(f'loaded results from {save_path}')
        result_list = json_data
    save_results(result_list, save_path)
    
    
if __name__ == "__main__":
    main()
