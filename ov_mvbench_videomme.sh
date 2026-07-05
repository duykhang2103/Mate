python -m tasks.eval.mvbench.ov_eval_mvbench \
       --pretrained_model_name_or_path MODELS/llava-onevision-7b \
       --save_path test_results/mvbench_ov_baseline_full \
       --num_frames 16 \
       --tasks "Action Sequence,Action Prediction,Moving Direction,Object Interaction,Unexpected Action" > log_mvbench_ov_baseline_full.log 2>&1


python -m tasks.eval.videomme.ov_eval_videomme \
       --pretrained_model_name_or_path MODELS/llava-onevision-7b \
       --save_path test_results/videomme_ov_long_100 \
       --num_frames 16 \
       --tasks "Long Video" \
       --max_samples 100 > log_videomme_ov_long_100.log 2>&1