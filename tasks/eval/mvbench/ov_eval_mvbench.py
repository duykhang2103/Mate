import logging
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torchvision

from decord import VideoReader, cpu
import transformers
import os

from tasks.eval.model_utils import load_llava_ov
from tasks.eval.mvbench import (
    MVBenchDataset,
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
        default='"./test_results/test_ov_mvbench"'
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
        default='eval_mvbench',
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
        help="Comma-separated list of task types to evaluate on (e.g. 'Action Sequence,State Change'). Default: all tasks.",
    )
    parser.add_argument(
        "--max_samples", 
        type=int,
        default=None,
        help="Max number of samples to evaluate per task. Default: all samples.",
    )
    parser.add_argument(
        "--use_vtp",
        action="store_true",
        help="Enable Visual Token Pruning.",
    )
    parser.add_argument(
        "--use_cluster_pruning",
        action="store_true",
        help="Use DPC-KNN cluster pruning instead of attention-based pruning.",
    )
    parser.add_argument(
        "--cluster_pruning_topk",
        type=float,
        default=0.4,
        help="Top-k ratio for cluster pruning.",
    )
    args = parser.parse_args()
    return args


def load_model_and_dataset(pretrained_model_name_or_path, num_frames, use_lora=False, 
                          lora_alpha=32, weight_dir=None, selected_layer=10, alpha=0.4, tau=0.8,
                          use_vtp=False, use_cluster_pruning=False, cluster_pruning_topk=0.4):
    """Load LLaVA-OneVision model and MVBench dataset."""
    model, processor = load_llava_ov(
        pretrained_model_name_or_path, 
        num_frames=num_frames,
        use_lora=use_lora,
        weight_dir=weight_dir,
        lora_alpha=lora_alpha,
        selected_layer=selected_layer,
        alpha=alpha,
        tau=tau,
        use_vtp=use_vtp,
        use_cluster_pruning=use_cluster_pruning,
        cluster_pruning_topk=cluster_pruning_topk,
    )
    logger.info('Done loading LLaVA-OneVision')
    # device_map="auto" in load_llava_ov already handles GPU placement
    # Do NOT call model.to() — it breaks device_map distribution

    dataset = MVBenchDataset(num_segments=num_frames)
    return model, processor, dataset


def infer_mvbench(
        args,
        model,
        processor,
        data_sample,
        conv_mode,
        pre_query_prompt=None,
        post_query_prompt=None,
        answer_prompt=None,
        return_prompt=None,
        print_res=False,
    ):
    """Run inference on a single MVBench sample using LLaVA-OV native interface."""
    video_list = data_sample["video_pils"]

    # Build question text with prompts
    question = data_sample['question']
    if pre_query_prompt:
        question = pre_query_prompt + question
    if post_query_prompt:
        question = question + post_query_prompt

    # Build conversation for LLaVA-OV using video format
    conversation = [
        {"role": "system", "content": "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question."},
        {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}
    ]
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

    # Handle answer_prompt/return_prompt for consistent formatting
    if answer_prompt and return_prompt:
        # Pre-fill answer_prompt if missing
        if not pred.startswith(answer_prompt):
            pred = answer_prompt + pred
        # Strip return_prompt from end if present
        if pred.endswith(return_prompt):
            pred = pred[:-len(return_prompt)]

    token_info = {}
    return pred, token_info


def run(args):
    """Main evaluation loop."""
    transformers.utils.logging.set_verbosity_error()

    print_res = False
    conv_mode = args.conv_mode
    pre_query_prompt = None
    post_query_prompt = "\nOnly give the best option."

    logger.info('Loading model and constructing dataset...')
    model, processor, dataset = load_model_and_dataset(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        num_frames=args.num_frames,
        use_lora=args.use_lora,
        lora_alpha=args.lora_alpha,
        weight_dir=args.weight_dir,
        selected_layer=args.selected_layer,
        alpha=args.alpha,
        tau=args.tau,
        use_vtp=args.use_vtp,
        use_cluster_pruning=args.use_cluster_pruning,
        cluster_pruning_topk=args.cluster_pruning_topk,
    )
    logger.info('Done loading model and dataset')

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
        
        pred, token_info = infer_mvbench(
            args,
            model,
            processor,
            example,
            conv_mode=conv_mode,
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
        
        tbar.update(len(result_list) - done_count)
        token_str = ""
        if hasattr(model, 'cache') and model.cache.num_tokens_after_prune is not None:
            token_str = f", Tokens: {model.cache.num_tokens_after_prune}"
        tbar.set_description_str(
            f"Task: {task_type}, Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100:.2f}%; "
            f"Total: {correct / total * 100:.2f}%{token_str}"
        )
        done_count = len(result_list)
    
    return result_list


def main():
    args = parse_args()
    save_path = args.save_path
    json_data = load_results(save_path)
    
    if json_data is None:
        logger.info(f'Started benchmarking, saving to: {save_path}')
        result_list = run(args)
        logger.info('Finished running')
    else:
        logger.info(f'Loaded results from {save_path}')
        result_list = json_data
    
    save_results(result_list, save_path)


if __name__ == "__main__":
    main()
