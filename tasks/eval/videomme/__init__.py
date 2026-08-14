import os
import json
import logging
import numpy as np
from tasks.eval.eval_utils import (
    dump_json,
    load_json,
    EvalDataset,
)


logger = logging.getLogger(__name__)


def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if not any([c in pred_option for c in 'abcdefgABCDEFG']):
        print(f"model doesn't follow instructions: {pred}")
    elif pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag

def save_results(result_list, save_path):
    if not result_list:
        print("WARNING: No results to save (0 samples processed). Check that video files exist on disk.")
        return
    final_res, acc_dict = {}, {}
    correct, total = 0, 0
    for res in result_list:
        task_type = res['task_type']
        if task_type not in acc_dict:
            acc_dict[task_type] = [0, 0] # correct, total
        acc_dict[task_type][1] += 1
        total += 1
        pred = res['pred']
        gt = res['gt']
        if check_ans(pred=pred, gt=gt):
            acc_dict[task_type][0] += 1
            correct += 1

    for k, v in acc_dict.items():
        final_res[k] = v[0] / v[1] * 100
    final_res['Avg'] = sum(v[0] for v in acc_dict.values()) / sum(v[1] for v in acc_dict.values()) * 100

    all_results = {
        "acc_dict": acc_dict,
        "result_list": result_list
    }
    dump_json(all_results, save_path, 'all_results.json')
    dump_json(final_res, save_path, 'upload_leaderboard.json')

    # Console logging
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for task_type, accuracy in final_res.items():
        if task_type != 'Avg':
            correct_count, total_count = acc_dict[task_type]
            print(f"{task_type}: {accuracy:.2f}% ({correct_count}/{total_count})")
    print("-"*50)
    print(f"Average Accuracy: {final_res['Avg']:.2f}%")
    print("="*50 + "\n")


def load_results(save_path):
    all_results = load_json(save_path, 'all_results.json')
    if all_results is not None and all_results.get('result_list'):
        result_list = all_results['result_list']
    else:
        result_list = None
    return result_list

class VideoMMEDataset(EvalDataset):
    data_list_info = {
        # "task_type (sub task name)": ("json file name", "image/video prefix", "data_type", "bound")
        "short": ("short.json", "DATAS/Video-MME/data", "video", False), # has start & end
        "medium": ("medium.json", "DATAS/Video-MME/data", "video", False), # has start & end
        "long": ("long.json", "DATAS/Video-MME/data", "video", False),
    }
    data_dir = "DATAS/Video-MME/json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        data_list_info = self.data_list_info
        data_dir = self.data_dir

        logger.info("Initializing VideoMMEDataset")
        logger.info("  cwd: %s", os.getcwd())
        logger.info("  data_dir: %s (exists=%s)", os.path.abspath(data_dir), os.path.exists(data_dir))

        self.data_list = []
        for k, v in data_list_info.items():
            json_path = os.path.join(data_dir, v[0])
            video_root = v[1]
            logger.info("  loading split=%s", k)
            logger.info("    json_path: %s (exists=%s)", os.path.abspath(json_path), os.path.exists(json_path))
            logger.info("    video_root: %s (exists=%s)", os.path.abspath(video_root), os.path.exists(video_root))

            with open(json_path, 'r') as f:
                json_data = json.load(f)
            logger.info("    samples loaded: %s", len(json_data))
            for data in json_data:
                self.data_list.append({
                    'task_type': k,
                    'prefix': video_root,
                    'data_type': v[2],
                    'bound': v[3],
                    'data': data
                })
        # self.data_list = self.data_list[:100] # for debug
        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
            'npy': self.read_npy,
        }

        logger.info("VideoMMEDataset total samples: %s", len(self.data_list))
                
        # # transform
        # crop_size = resolution
        # scale_size = resolution
        # input_mean = [0.48145466, 0.4578275, 0.40821073]
        # input_std = [0.26862954, 0.26130258, 0.27577711]
        # self.transform = T.Compose([
        #     GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        #     GroupCenterCrop(crop_size),
        #     Stack(),
        #     ToTorchFormatTensor(),
        #     GroupNormalize(input_mean, input_std) 
        # ])
    
    def __getitem__(self, idx):
        try:
            question, answer = self.qa_template(self.data_list[idx]['data'])
            task_type = self.data_list[idx]['task_type']
            decord_method = self.decord_method[self.data_list[idx]['data_type']]
            bound = None
            if self.data_list[idx]['bound']:
                bound = (
                    self.data_list[idx]['data']['start'],
                    self.data_list[idx]['data']['end'],
                )
            video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])

            # Trace video path and existence before attempting to read
            logger.debug("__getitem__ idx=%s task=%s video=%s", idx, task_type, video_path)
            if not os.path.exists(video_path):
                logger.warning("Video file does not exist: %s (idx=%s, task=%s)", video_path, idx, task_type)
                return None

            try:
                images_group = decord_method(video_path, bound)
            except Exception:
                logger.exception("Failed to read video %s (idx=%s, task=%s)", video_path, idx, task_type)
                return None

            return {
                'video_path': video_path, 
                'video_pils': images_group,
                'question': question, 
                'answer': answer,
                'task_type': task_type,
            }
        except IndexError:
            raise
        except Exception:
            return None
        

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

