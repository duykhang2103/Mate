#!/usr/bin/env bash

set -euo pipefail

if [ "${SAFEPRUNEVID_ALLOW_ARCHIVED:-0}" != "1" ]; then
    printf '%s\n' \
        "SafePruneVid is archived and no longer an active experiment." \
        "For reproduction only, set SAFEPRUNEVID_ALLOW_ARCHIVED=1." >&2
    exit 2
fi

# Authentication, when required, is read from the environment. Never put a
# Hugging Face credential in this archived runner or print it to a log.
STAGE="${SAFEPRUNEVID_STAGE:-calibrate}"
MODEL_DIR="${PLLAVA_MODEL_DIR:-MODELS/pllava-7b}"
RESULT_ROOT="${SAFEPRUNEVID_RESULT_ROOT:-results/safeprunevid}"
CALIBRATION_ROOT="${RESULT_ROOT}/calibration"
QUERY_GATE_JSON="${QUERY_GATE_JSON:-${RESULT_ROOT}/query_kill_gate.json}"
ADBR_M_JSON="${RESULT_ROOT}/adbr_m_calibration.json"
ADBR_E_JSON="${RESULT_ROOT}/adbr_e_calibration.json"
ORIGINAL_RESULT="${ORIGINAL_RESULT:-${RESULT_ROOT}/original_timed/all_results.json}"
QUERY_RESULT="${QUERY_RESULT:-zzresults/Video-MME/results/pllava_videomme_query_merge_w07_r075_full/pllava_videomme_query_merge_w07_r075_full/all_results.json}"
NOVELTY_RESULT="${NOVELTY_RESULT:-results/pllava_videomme_novelty_w00_r075_full/all_results.json}"

COMMON_VIDEOMME_ARGS=(
    --pretrained_model_name_or_path "${MODEL_DIR}"
    --num_frames 16
    --use_lora
    --lora_alpha 14
    --conv_mode eval_videomme
    --alpha 0.4
    --selected_layer 10
    --tau 0.8
    --temporal_segment_ratio 0.25
    --cluster_ratio 0.5
)

pip install "huggingface-hub==0.21.4"
pip install "transformers==4.46.3"

query_router_args=()
if [ -f "${QUERY_GATE_JSON}" ]; then
    retain_query="$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["retain_query"]))' "${QUERY_GATE_JSON}")"
    if [ "${retain_query}" = "1" ]; then
        query_router_args=(
            --distortion_use_query_relevance
            --distortion_query_weight 0.7
        )
    fi
fi

read_epsilon() {
    python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["epsilon"])' "$1"
}

run_videomme_adbr() {
    local epsilon="$1"
    local save_path="$2"
    shift 2
    python -m tasks.eval.videomme.pllava_eval_videomme \
        "${COMMON_VIDEOMME_ARGS[@]}" \
        --save_path "${save_path}" \
        --use_distortion_routing \
        --distortion_epsilon "${epsilon}" \
        "${query_router_args[@]}" \
        "$@"
}

case "${STAGE}" in
    query-gate)
        python scripts/evaluate_query_kill_gate.py \
            --novelty-only "${NOVELTY_RESULT}" \
            --query-guided "${QUERY_RESULT}" \
            --json-output "${QUERY_GATE_JSON}"
        ;;

    baseline-timing)
        python -m tasks.eval.videomme.pllava_eval_videomme \
            "${COMMON_VIDEOMME_ARGS[@]}" \
            --save_path "${RESULT_ROOT}/original_timed"
        ;;

    epsilon-0029)
        run_videomme_adbr \
            0.029 \
            "${CALIBRATION_ROOT}/epsilon_0p029" \
            --max_videos_per_split 100
        ;;

    calibrate)
        epsilon_grid=(0.0025 0.005 0.01 0.02 0.04 0.08)
        calibration_results=()
        for epsilon in "${epsilon_grid[@]}"; do
            epsilon_name="${epsilon//./p}"
            save_path="${CALIBRATION_ROOT}/epsilon_${epsilon_name}"
            run_videomme_adbr \
                "${epsilon}" \
                "${save_path}" \
                --max_videos_per_split 100
            calibration_results+=("${save_path}/all_results.json")
        done
        python scripts/select_distortion_epsilon.py \
            "${calibration_results[@]}" \
            --target-tflops 8.9714 \
            --mode match \
            --tolerance 0.01 \
            --required-videos-per-group 100 \
            --json-output "${ADBR_M_JSON}"
        python scripts/select_distortion_epsilon.py \
            "${calibration_results[@]}" \
            --target-tflops 8.5181 \
            --mode at-most \
            --required-videos-per-group 100 \
            --json-output "${ADBR_E_JSON}"
        ;;

    full-m)
        run_videomme_adbr \
            "$(read_epsilon "${ADBR_M_JSON}")" \
            "${RESULT_ROOT}/adbr_m_full"
        ;;

    full-e)
        run_videomme_adbr \
            "$(read_epsilon "${ADBR_E_JSON}")" \
            "${RESULT_ROOT}/adbr_e_full"
        ;;

    compare)
        for variant in adbr_m_full adbr_e_full; do
            python scripts/compare_paired_results.py \
                --baseline "${ORIGINAL_RESULT}" \
                --candidate "${RESULT_ROOT}/${variant}/all_results.json" \
                --json-output "${RESULT_ROOT}/${variant}/paired_vs_original.json"
            python scripts/evaluate_acceptance_gate.py \
                --baseline "${ORIGINAL_RESULT}" \
                --candidate "${RESULT_ROOT}/${variant}/all_results.json" \
                --json-output "${RESULT_ROOT}/${variant}/acceptance_gate.json"
        done
        ;;

    router)
        candidate_variant="${ROUTER_CANDIDATE_VARIANT:-adbr_m_full}"
        python scripts/train_safety_router.py \
            --original "${ORIGINAL_RESULT}" \
            --candidate "${RESULT_ROOT}/${candidate_variant}/all_results.json" \
            --model-output "${RESULT_ROOT}/safety_router/model.json" \
            --routed-results-output "${RESULT_ROOT}/safety_router/test_results.json" \
            --report-output "${RESULT_ROOT}/safety_router/report.json"
        ;;

    routed-full)
        candidate_variant="${ROUTER_CANDIDATE_VARIANT:-adbr_m_full}"
        calibration_json="${ADBR_M_JSON}"
        if [ "${candidate_variant}" = "adbr_e_full" ]; then
            calibration_json="${ADBR_E_JSON}"
        fi
        run_videomme_adbr \
            "$(read_epsilon "${calibration_json}")" \
            "${RESULT_ROOT}/safety_router/full_runtime" \
            --safety_router_path "${RESULT_ROOT}/safety_router/model.json"
        ;;

    mvbench)
        variant="${ADBR_VARIANT:-m}"
        calibration_json="${ADBR_M_JSON}"
        if [ "${variant}" = "e" ]; then
            calibration_json="${ADBR_E_JSON}"
        fi
        epsilon="$(read_epsilon "${calibration_json}")"
        python -m tasks.eval.mvbench.pllava_eval_mvbench \
            --pretrained_model_name_or_path "${MODEL_DIR}" \
            --weight_dir "${MODEL_DIR}" \
            --save_path "${RESULT_ROOT}/mvbench_original" \
            --num_frames 16 --use_lora --lora_alpha 14 \
            --pooling_shape 16-12-12 --selected_layer 10 \
            --alpha 0.4 --tau 0.8 \
            --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
            --conv_mode eval_mvbench --max_new_tokens 100
        python -m tasks.eval.mvbench.pllava_eval_mvbench \
            --pretrained_model_name_or_path "${MODEL_DIR}" \
            --weight_dir "${MODEL_DIR}" \
            --save_path "${RESULT_ROOT}/mvbench_adbr_${variant}" \
            --num_frames 16 --use_lora --lora_alpha 14 \
            --pooling_shape 16-12-12 --selected_layer 10 \
            --alpha 0.4 --tau 0.8 \
            --temporal_segment_ratio 0.25 --cluster_ratio 0.5 \
            --conv_mode eval_mvbench --max_new_tokens 100 \
            --use_distortion_routing \
            --distortion_epsilon "${epsilon}" \
            "${query_router_args[@]}"
        ;;

    *)
        echo "Unknown SAFEPRUNEVID_STAGE=${STAGE}" >&2
        echo "Use query-gate, baseline-timing, calibrate, full-m, full-e, compare, router, routed-full, or mvbench." >&2
        exit 2
        ;;
esac
