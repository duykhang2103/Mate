
import torch
import os
from peft import get_peft_model, LoraConfig, TaskType
from safetensors import safe_open
from peft import PeftModel
from tasks.eval.eval_utils import Conversation
from models.pllava import PllavaProcessor, PllavaForConditionalGeneration, PllavaConfig
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map,load_checkpoint_in_model
from accelerate.utils import get_balanced_memory
from transformers import LlavaOnevisionForConditionalGeneration
# from mmcv.runner import load_checkpoint
load_checkpoint = None
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


from transformers import StoppingCriteria
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
            return False
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len:], skip_special_tokens=True
            )
            flag = True
            for output in outputs:
                for keyword in self.keywords:
                    if keyword not in output:
                        flag = False
                        return False
            return flag

def load_llava_next_video(repo_id, num_frames, use_lora=False, weight_dir=None, lora_alpha=32, use_multi_gpus=False, pooling_shape=(16,12,12)):
    kwargs = {
        'num_frames': num_frames,
    }
    # print("===============>pooling_shape", pooling_shape)
    if num_frames == 0:
        kwargs.update(pooling_shape=(0,12,12)) # produce a bug if ever usen the pooling projector
    
    if 'LLaVA-NeXT-Video' in repo_id:
        config = LlavaNextVideoConfig.from_pretrained(
            repo_id,
            # repo_id if not use_lora else weight_dir,
            # pooling_shape=pooling_shape,
            **kwargs,
        )
        with torch.no_grad():
            model = LlavaNextVideoForConditionalGeneration.from_pretrained(repo_id, config=config, torch_dtype=torch.bfloat16)
        try:
            processor = LlavaNextVideoProcessor.from_pretrained(repo_id)
        except Exception as e:
            processor = LlavaNextVideoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
        
        logger.info("Loading optical flow model")
        try:
            flow_checkpoint = 'raft_8x2_100k_mixed_368x768.pth'
            checkpoint = load_checkpoint(model.optical_flow_model, flow_checkpoint, map_location='cpu')
            logger.info("Successful loading optical flow model")
        except:
            logger.info("Loading optical flow model failed, use default model")

    elif 'tarsier' in repo_id:
        try:
            config = LlavaConfig.from_pretrained(
                repo_id if not use_lora else weight_dir,
                pooling_shape=pooling_shape,
                **kwargs,
            )
        except:
            config = LlavaConfig.from_pretrained(
                repo_id,
                pooling_shape=pooling_shape,
                **kwargs,
            )
        with torch.no_grad():
            model = TarsierForConditionalGeneration.from_pretrained(repo_id, config=config, torch_dtype=torch.bfloat16)
        
        try:
            processor = LlavaProcessor.from_pretrained(repo_id)
        except Exception as e:
            processor = LlavaProcessor.from_pretrained('MODELS/tarsier')
    
    else:
        raise ValueError("Invalid repo id")

    # config lora
    if use_lora and weight_dir is not None:
        print("Use lora")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,  target_modules=["q_proj", "v_proj"],
            r=128, lora_alpha=lora_alpha, lora_dropout=0.
        )
        print("Lora Scaling:", lora_alpha/128)
        model.language_model = get_peft_model(model.language_model, peft_config)
        assert weight_dir is not None, "pass a folder to your lora weight"
        print("Finish use lora")
    
    # load weights
    if weight_dir is not None:
        state_dict = {}
        save_fnames = os.listdir(weight_dir)
        if "model.safetensors" in save_fnames:
            use_full = False
            for fn in save_fnames:
                if fn.startswith('model-0'):
                    use_full=True        
                    break
        else:
            use_full= True

        if not use_full:
            print("Loading weight from", weight_dir, "model.safetensors")
            with safe_open(f"{weight_dir}/model.safetensors", framework="pt", device="cpu") as f:
                for k in f.keys():
                    print("model.safetensors", k)
                    state_dict[k] = f.get_tensor(k)
        else:
            print("Loading weight from", weight_dir)
            for fn in save_fnames:
                if fn.startswith('model-0'):
                    with safe_open(f"{weight_dir}/{fn}", framework="pt", device="cpu") as f:
                        for k in f.keys():
                            k_new = k
                            # if 'language_model.model' in k:
                            #     k_new = k.replace('language_model.model', 'language_model.base_model.model.model')
                            state_dict[k_new] = f.get_tensor(k)
            
        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
        print('loading state:', msg)
        # print('pretrained keys:', list(state_dict.keys()))
        # print('model keys:', model.state_dict().keys())

    # dispatch model weight
    if use_multi_gpus:
        max_memory = get_balanced_memory(
            model,
            max_memory=None,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype='bfloat16',
            low_zero=False,
        )

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype='bfloat16'
        )

        dispatch_model(model, device_map=device_map)
        print(model.hf_device_map)

    model = model.eval()

    return model, processor

def load_pllava(repo_id, num_frames, use_lora=False, weight_dir=None, lora_alpha=32, use_multi_gpus=False, pooling_shape=(16,12,12),selected_layer=10, alpha=0.1, head=0, softmax=1.0, tau=1.0, cluster_ratio=1.0, temporal_segment_ratio=1.0, use_entropy_adaptive=False, tau_entropy=0.8, entropy_fallback_layer=20, use_motion_adaptive=False, motion_scale=0.5, motion_invert=False, use_borderline_preservation=False, borderline_margin=0.1, use_weighted_merge=True, use_flow_pruning=False, flow_dynamic_ratio=0.5, use_cluster_pruning=False, cluster_pruning_topk=0.4):
    kwargs = {
        'num_frames': num_frames,
    }
    # print("===============>pooling_shape", pooling_shape)
    if num_frames == 0:
        kwargs.update(pooling_shape=(0,12,12)) # produce a bug if ever usen the pooling projector
    
    if 'llava' in repo_id:
        config = PllavaConfig.from_pretrained(
            repo_id if not use_lora else weight_dir,
            pooling_shape=pooling_shape,
            selected_layer=selected_layer,
            alpha=alpha,
            head=head,
            softmax=softmax,
            tau=tau,
            cluster_ratio=cluster_ratio,
            temporal_segment_ratio=temporal_segment_ratio,
            use_entropy_adaptive=use_entropy_adaptive,
            tau_entropy=tau_entropy,
            entropy_fallback_layer=entropy_fallback_layer,
            use_motion_adaptive=use_motion_adaptive,
            motion_scale=motion_scale,
            motion_invert=motion_invert,
            use_borderline_preservation=use_borderline_preservation,
            borderline_margin=borderline_margin,
            use_weighted_merge=use_weighted_merge,
            use_flow_pruning=use_flow_pruning,
            flow_dynamic_ratio=flow_dynamic_ratio,
            use_cluster_pruning=use_cluster_pruning,
            cluster_pruning_topk=cluster_pruning_topk,
            **kwargs,
        )
        with torch.no_grad():
            # model = PllavaFlowForConditionalGeneration.from_pretrained(repo_id, config=config, torch_dtype=torch.bfloat16)
            model = PllavaForConditionalGeneration.from_pretrained(repo_id, config=config, torch_dtype=torch.bfloat16)
        try:
            processor = PllavaProcessor.from_pretrained(repo_id)
        except Exception as e:
            processor = PllavaProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
        
        logger.info("Loading optical flow model")
        try:
            flow_checkpoint = 'raft_8x2_100k_mixed_368x768.pth'
            checkpoint = load_checkpoint(model.optical_flow_model, flow_checkpoint, map_location='cpu')
            logger.info("Successful loading optical flow model")
        except:
            logger.info("Loading optical flow model failed, use default model")

    elif 'tarsier' in repo_id:
        try:
            config = LlavaConfig.from_pretrained(
                repo_id if not use_lora else weight_dir,
                pooling_shape=pooling_shape,
                selected_layer=selected_layer,
                alpha=alpha,
                head=head,
                softmax=softmax,
                **kwargs,
            )
        except:
            config = LlavaConfig.from_pretrained(
                repo_id,
                pooling_shape=pooling_shape,
                selected_layer=selected_layer,
                alpha=alpha,
                head=head,
                softmax=softmax,
                **kwargs,
            )
        with torch.no_grad():
            model = TarsierForConditionalGeneration.from_pretrained(repo_id, config=config, torch_dtype=torch.bfloat16)
        
        try:
            processor = LlavaProcessor.from_pretrained(repo_id)
        except Exception as e:
            processor = LlavaProcessor.from_pretrained('MODELS/tarsier')
    
    else:
        raise ValueError("Invalid repo id")

    # config lora
    if use_lora and weight_dir is not None:
        print("Use lora")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,  target_modules=["q_proj", "v_proj"],
            r=128, lora_alpha=lora_alpha, lora_dropout=0.
        )
        print("Lora Scaling:", lora_alpha/128)
        model.language_model = get_peft_model(model.language_model, peft_config)
        assert weight_dir is not None, "pass a folder to your lora weight"
        print("Finish use lora")
    
    # load weights
    if weight_dir is not None:
        state_dict = {}
        save_fnames = os.listdir(weight_dir)
        if "model.safetensors" in save_fnames:
            use_full = False
            for fn in save_fnames:
                if fn.startswith('model-0'):
                    use_full=True        
                    break
        else:
            use_full= True

        if not use_full:
            print("Loading weight from", weight_dir, "model.safetensors")
            with safe_open(f"{weight_dir}/model.safetensors", framework="pt", device="cpu") as f:
                for k in f.keys():
                    print("model.safetensors", k)
                    state_dict[k] = f.get_tensor(k)
        else:
            print("Loading weight from", weight_dir)
            for fn in save_fnames:
                if fn.startswith('model-0'):
                    with safe_open(f"{weight_dir}/{fn}", framework="pt", device="cpu") as f:
                        for k in f.keys():
                            # print(fn, k)
                            state_dict[k] = f.get_tensor(k)
            
        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
        print('model load state:', msg)
    # dispatch model weight
    if use_multi_gpus:
        max_memory = get_balanced_memory(
            model,
            max_memory=None,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype='bfloat16',
            low_zero=False,
        )

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["LlamaDecoderLayer"],
            dtype='bfloat16'
        )

        dispatch_model(model, device_map=device_map)
        print(model.hf_device_map)

    model = model.eval()

    return model, processor


def load_llava_ov(pretrained_model_name_or_path, num_frames=16, use_lora=False, weight_dir=None, lora_alpha=32, 
                  selected_layer=10, alpha=0.4, tau=0.8, use_vtp=True,
                  use_cluster_pruning=False, cluster_pruning_topk=0.4,
                  **kwargs):
    """Load LLaVA-OneVision with optional VTP support."""
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    from models.llava_ov.qwen2_vtp import Qwen2ModelVTP
    from models.llava_ov.elastic_cache_ov import ElasticCacheOV
    
    logger.info(f"Loading LLaVA-OneVision from {pretrained_model_name_or_path}")
    
    # Always load base model first (device_map="auto" works with this)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Required for attention weights
        **kwargs
    )
    
    if use_vtp:
        # Create pruning cache
        image_token_id = getattr(model.config, 'image_token_index', 151646)
        video_token_id = getattr(model.config, 'video_token_index', 151647)
        cache = ElasticCacheOV(
            alpha=alpha,
            total_num_layers=model.config.text_config.num_hidden_layers,
            selected_layer=selected_layer,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            use_cluster_pruning=use_cluster_pruning,
            cluster_pruning_topk=cluster_pruning_topk,
        )
        
        # Replace Qwen2Model with Qwen2ModelVTP
        original_model = model.language_model.model
        model.language_model.model = Qwen2ModelVTP(original_model, cache)
        
        # Store cache reference on top-level model for token counting
        model.cache = cache
        
        # Use forward pre-hook to capture input_ids before parent converts them to inputs_embeds
        def _capture_input_ids(module, args, kwargs):
            if 'input_ids' in kwargs and kwargs['input_ids'] is not None:
                module.cache._input_ids = kwargs['input_ids']
            elif len(args) > 0:
                module.cache._input_ids = args[0]
        model.register_forward_pre_hook(_capture_input_ids, with_kwargs=True)
        
        logger.info(f"Loaded with VTP (alpha={alpha}, layer={selected_layer}, cluster={use_cluster_pruning})")
    else:
        logger.info("Loaded base model (no VTP)")
    
    # Explicitly cast vision tower to bfloat16 to fix dtype mismatch
    for name, param in model.vision_tower.named_parameters():
        if param.is_floating_point():
            param.data = param.data.to(torch.bfloat16)
    for name, buf in model.vision_tower.named_buffers():
        if buf.is_floating_point():
            buf.data = buf.data.to(torch.bfloat16)
    
    # Load processor
    try:
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path)
    except Exception as e:
        logger.warning(f"Failed to load processor: {e}, trying fallback")
        processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
    
    model = model.eval()
    
    return model, processor


def load_adapters(model, adapter_model_name_or_paths):

    for adapter_model_name_or_path in adapter_model_name_or_paths:
        if not isinstance(model, PeftModel):
            model = PeftModel.from_pretrained(model, adapter_model_name_or_path, adapter_model_name_or_path)
        else:
            model.load_adapter(adapter_model_name_or_path, adapter_model_name_or_path)

    return model


def pllava_answer(conv: Conversation, model, processor, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, stop_criteria_keywords=None, print_res=False):
    # torch.cuda.empty_cache()
    prompt = conv.get_prompt()
    inputs = processor(text=prompt, images=img_list, return_tensors="pt")
    if inputs['pixel_values'] is None:
        inputs.pop('pixel_values')
    inputs = inputs.to(model.device)
    
    # set up stopping criteria
    if stop_criteria_keywords is not None:
        stopping_criteria = [KeywordsStoppingCriteria(stop_criteria_keywords, processor.tokenizer, inputs["input_ids"])]
    else:
        stopping_criteria= None
    with torch.no_grad():
        try:
            output_token = model.generate(**inputs, media_type='video',
                                        do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams, min_length=min_length, 
                                        top_p=top_p, repetition_penalty=repetition_penalty, length_penalty=length_penalty, temperature=temperature, 
                                        stopping_criteria=stopping_criteria,use_cache=True)
        except:
            output_token = model.generate(**inputs,
                                        do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams, min_length=min_length, 
                                        top_p=top_p, repetition_penalty=repetition_penalty, length_penalty=length_penalty, temperature=temperature, 
                                        stopping_criteria=stopping_criteria,use_cache=True)
        output_text = processor.batch_decode(output_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if print_res: # debug usage
        print('### PROMPTING LM WITH: ', prompt)
        print('### LM OUTPUT TEXT:  ', output_text)
    if conv.roles[-1] == "<|im_start|>assistant\n":
        split_tag = "<|im_start|> assistant\n"
    else:
        split_tag = conv.roles[-1]
    output_text = output_text.split(split_tag)[-1]
    ending = conv.sep if isinstance(conv.sep, str) else conv.sep[1]
    output_text = output_text.removesuffix(ending).strip()
    conv.messages[-1][1] = output_text

    # Extract token counts for FLOPs computation
    token_info = {}
    try:
        # Vision token counts are on the top-level model (PllavaForConditionalGeneration)
        token_info['raw_vision_tokens'] = getattr(model, '_last_raw_vision_tokens', None)
        token_info['merged_vision_tokens'] = getattr(model, '_last_merged_vision_tokens', None)
        # LLM pruned token count is on the inner model (LlamaModelVTP)
        # With LoRA: PeftModel -> LlamaForCausalLMVTP -> LlamaModelVTP
        # Without LoRA: LlamaForCausalLMVTP -> LlamaModelVTP
        lm_model = model.language_model.model
        if not hasattr(lm_model, 'last_pruned_token_count'):
            lm_model = lm_model.model  # PeftModel wraps one extra level
        token_info['pruned_tokens'] = getattr(lm_model, 'last_pruned_token_count', None)
        # Use dynamic layer if available, otherwise fixed layer
        dynamic_layer = getattr(lm_model, 'last_dynamic_selected_layer', None)
        token_info['pruning_layer'] = dynamic_layer if dynamic_layer is not None else getattr(lm_model, 'selected_layer', None)
    except Exception:
        pass

    return output_text, conv, token_info

def ov_answer(conv: Conversation, model, processor, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, stop_criteria_keywords=None, print_res=False):
    """Answer function for LLaVA-OneVision using native apply_chat_template."""
    # Build conversation for LLaVA-OV's chat template
    # conv.messages[-1][0] is the user message (question)
    user_msg = conv.messages[-1][0]
    conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_msg}]}]

    text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=img_list, return_tensors="pt")
    # Move to device, only cast float tensors to bfloat16
    inputs = {k: v.to(model.device) if not v.is_floating_point() else v.to(model.device, dtype=torch.bfloat16) for k, v in inputs.items()}

    if stop_criteria_keywords is not None:
        stopping_criteria = [KeywordsStoppingCriteria(stop_criteria_keywords, processor.tokenizer, inputs["input_ids"])]
    else:
        stopping_criteria = None

    with torch.no_grad():
        output_token = model.generate(**inputs,
                                      do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams, min_length=min_length,
                                      top_p=top_p, repetition_penalty=repetition_penalty, length_penalty=length_penalty, temperature=temperature,
                                      stopping_criteria=stopping_criteria, use_cache=True)
        output_text = processor.batch_decode(output_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if print_res:
        print('### PROMPTING LM WITH: ', text)
        print('### LM OUTPUT TEXT:  ', output_text)

    # Extract just the assistant response
    if "assistant\n" in output_text:
        output_text = output_text.split("assistant\n")[-1]
    output_text = output_text.strip()

    conv.messages[-1][1] = output_text

    token_info = {}
    return output_text, conv, token_info


def llava_next_video_answer(conv: Conversation, model, processor, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, stop_criteria_keywords=None, print_res=False):
    # torch.cuda.empty_cache()
    prompt = conv.get_prompt()
    inputs = processor(text=prompt, videos=img_list, return_tensors="pt")
    # if inputs['pixel_values'] is None:
    #     inputs.pop('pixel_values')
    inputs = inputs.to(model.device)
    # set up stopping criteria
    
    if stop_criteria_keywords is not None:
        stopping_criteria = [KeywordsStoppingCriteria(stop_criteria_keywords, processor.tokenizer, inputs["input_ids"])]
    else:
        stopping_criteria= None

    with torch.no_grad():
        output_token = model.generate(**inputs,
                                      do_sample=do_sample, max_new_tokens=max_new_tokens, num_beams=num_beams, min_length=min_length, 
                                      top_p=top_p, repetition_penalty=repetition_penalty, length_penalty=length_penalty, temperature=temperature, 
                                      stopping_criteria=stopping_criteria,)
        output_text = processor.batch_decode(output_token, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if print_res: # debug usage
        print('### PROMPTING LM WITH: ', prompt)
        print('### LM OUTPUT TEXT:  ', output_text)
    if conv.roles[-1] == "<|im_start|>assistant\n":
        split_tag = "<|im_start|> assistant\n"
    else:
        split_tag = conv.roles[-1]
    output_text = output_text.split(split_tag)[-1]
    ending = conv.sep if isinstance(conv.sep, str) else conv.sep[1]
    output_text = output_text.removesuffix(ending).strip()
    conv.messages[-1][1] = output_text
    return output_text, conv