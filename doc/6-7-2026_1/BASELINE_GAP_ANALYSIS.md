# Baseline Gap Investigation (52.42% vs 58%)

> **Date**: 2026-07-06
> **Issue**: LLaVA-OV MVBench baseline shows 52.42%, PruneVid paper reports 58%

---

## Root Cause: 3 Bugs in `ov_eval_mvbench.py`

### Bug 1: Missing System Prompt (estimated impact: ~3-5%)

`infer_mvbench()` at line 179-180 constructs the prompt WITHOUT the MVBench system prompt:

```python
conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}]
text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
```

The system prompt from `eval_utils.py:164` is never included:
```
Carefully watch the video and pay attention to the cause and sequence of events,
the detail and movement of objects, and the action and pose of persons.
Based on your observations, select the best option that accurately addresses the question.
```

**Fix**: Add system message to conversation:
```python
conversation = [
    {"role": "system", "content": "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question."},
    {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": question}]}
]
```

### Bug 2: Unused `answer_prompt`/`return_prompt` (estimated impact: ~1-2%)

`infer_mvbench()` receives `answer_prompt="Best option:("` and `return_prompt='('` (line 273-274) but NEVER uses them. The PLLaVA eval uses these to:
1. Pre-fill `"Best option:("` to guide model output format
2. Strip it from output and add `"("` back

Without this, the model outputs freeform text instead of `(A)` / `(B)` format, making answer matching less reliable.

**Fix**: Add answer_prompt pre-fill and return_prompt stripping to `infer_mvbench()`.

### Bug 3: Missing Video Files (214/4000 samples)

| Task | Missing | Issue |
|------|---------|-------|
| Fine-grained Pose | 200/200 | `nturgbd/` directory missing entirely |
| Action Sequence | 12/200 | 12 files missing in `star/Charades_v1_480/` |
| Object Existence | 2/200 | 2 files missing in `clevrer/video_validation/` |

These are silently skipped during eval, reducing effective sample count.

---

## Priority Fix Order

1. **Add system prompt** — biggest impact, quick fix
2. **Re-extract missing video data** — download/extract `nturgbd`, star, clevrer zips
3. **Add answer_prompt pre-fill** — format guidance for cleaner outputs
4. **Re-run full 20-task baseline** — should hit ~57-58%
5. **Then run VTP comparison** — against corrected baseline

---

## Expected Outcome After Fixes

- Baseline: ~57-58% (matching PruneVid)
- VTP alpha=0.4: should beat baseline by 1-3%
- This gives a publishable result
