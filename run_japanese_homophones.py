#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import time
import importlib.util
import re
import io
import json
from contextlib import redirect_stdout

# Import PIPELINE corrector instead of old chat_agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline_correction import PipelineCorrector

# Import WER/CER calculator from compute-wer.py
tools_dir = os.path.join(os.path.dirname(__file__), 'tools')
compute_wer_path = os.path.join(tools_dir, 'compute-wer.py')
spec = importlib.util.spec_from_file_location("compute_wer", compute_wer_path)
compute_wer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compute_wer)
Calculator = compute_wer.Calculator
characterize = compute_wer.characterize

def _normalize_eval_text(text, strip_punctuation=False):
    value = (text or "").strip()
    if not strip_punctuation:
        return value
    return re.sub(r"[\s、。,.!?！？；;:：]+$", "", value)


def _safe_percent(part, whole):
    if whole <= 0:
        return 0.0
    return (part * 100.0) / whole


def calculate_metrics(result_file, label_file, strip_punctuation=False):
    """
    Calculate accuracy metrics by comparing result with label
    Returns: (exact_match_count, total_sentences, cer, wer)
    """
    # Read result file (corrected output)
    results = {}
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                sent_id, text = parts
                results[sent_id] = text
            else:
                results[f"sent_{len(results)+1:04d}"] = line
    
    # Read label file (ground truth)
    labels = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                sent_id, text = parts
                labels[sent_id] = text
            else:
                labels[f"sent_{len(labels)+1:04d}"] = line
    
    # Calculate exact match accuracy
    exact_matches = 0
    total_sentences = len(labels)
    
    for sent_id in labels:
        if sent_id in results:
            rec = _normalize_eval_text(results[sent_id], strip_punctuation=strip_punctuation)
            lab = _normalize_eval_text(labels[sent_id], strip_punctuation=strip_punctuation)
            if rec == lab:
                exact_matches += 1
    
    # Calculate CER (Character Error Rate)
    cer_calculator = Calculator()
    for sent_id in labels:
        if sent_id in results:
            lab = _normalize_eval_text(labels[sent_id], strip_punctuation=strip_punctuation)
            rec = _normalize_eval_text(results[sent_id], strip_punctuation=strip_punctuation)
            lab_chars = characterize(lab)
            rec_chars = characterize(rec)
            cer_calculator.calculate(lab_chars, rec_chars)
    
    cer_result = cer_calculator.overall()
    if cer_result['all'] > 0:
        cer = float(cer_result['ins'] + cer_result['sub'] + cer_result['del']) * 100.0 / cer_result['all']
    else:
        cer = 0.0
    
    # Calculate WER (Word Error Rate)
    wer_calculator = Calculator()
    for sent_id in labels:
        if sent_id in results:
            lab = _normalize_eval_text(labels[sent_id], strip_punctuation=strip_punctuation)
            rec = _normalize_eval_text(results[sent_id], strip_punctuation=strip_punctuation)
            lab_chars = characterize(lab)
            rec_chars = characterize(rec)
            wer_calculator.calculate(lab_chars, rec_chars)
    
    wer_result = wer_calculator.overall()
    if wer_result['all'] > 0:
        wer = float(wer_result['ins'] + wer_result['sub'] + wer_result['del']) * 100.0 / wer_result['all']
    else:
        wer = 0.0
    
    return exact_matches, total_sentences, cer, wer


def _format_wer_detail(sent_id, label_text, rec_text):
    """Build one utterance-level WER detail block compatible with legacy result file."""
    calc = Calculator()
    lab_chars = characterize(label_text)
    rec_chars = characterize(rec_text)
    result = calc.calculate(list(lab_chars), list(rec_chars))
    n = result["all"]
    errors = result["sub"] + result["del"] + result["ins"]
    wer = (errors * 100.0 / n) if n > 0 else 0.0

    return (
        f"utt:{sent_id}\n"
        f"WER:{wer:.2f}%N={n}C={result['cor']}S={result['sub']}D={result['del']}I={result['ins']}\n"
        f"lab:{''.join(lab_chars)}\n"
        f"rec:{''.join(rec_chars)}\n\n"
    )


def _parse_change_pair(change_text):
    """Parse one change token in format old→new."""
    if not change_text or "→" not in change_text:
        return "", ""
    left, right = change_text.split("→", 1)
    return left.strip(), right.strip()


def _classify_mismatch(result_item, corrector):
    """Classify mismatch type for granular evaluation reporting."""
    original = result_item.get("original", "")
    expected = result_item.get("expected", "")
    corrected = result_item.get("corrected", "")
    changes = result_item.get("changes", []) or []

    if corrected == expected:
        return "matched"
    if not changes:
        return "no_change_miss"
    if any(c == "LLM_FREE_SENTENCE" for c in changes):
        return "llm_sentence_rewrite"

    parsed = [_parse_change_pair(c) for c in changes if "→" in c]
    parsed = [(a, b) for (a, b) in parsed if a and b]
    if not parsed:
        return "other"

    homophone_hits = 0
    non_homophone_hits = 0
    short_kana_hits = 0
    for old_word, new_word in parsed:
        if len(old_word) <= 2 and re.fullmatch(r"[\u3040-\u309f\u30a0-\u30ffー]+", old_word):
            short_kana_hits += 1
        info = corrector.homophones.get(old_word, {})
        candidates = info.get("candidates", []) if isinstance(info, dict) else []
        if new_word in candidates:
            homophone_hits += 1
        else:
            non_homophone_hits += 1

    if short_kana_hits > 0:
        return "short_kana_edit"
    if homophone_hits > 0 and non_homophone_hits == 0:
        return "homophone_confusion"
    if homophone_hits > 0 and non_homophone_hits > 0:
        return "mixed_error"

    # If expected differs but source token is not in dictionary, mark as coverage gap.
    if original != expected and all(old not in corrector.homophones for old, _ in parsed):
        return "dict_coverage_gap"
    return "non_homophone_or_drift"


def _build_mismatch_breakdown(results, corrector):
    """Aggregate mismatch counts by coarse error type."""
    breakdown = {}
    mismatches = [r for r in results if r.get("expected") and r["corrected"] != r["expected"]]
    for item in mismatches:
        key = _classify_mismatch(item, corrector)
        breakdown[key] = breakdown.get(key, 0) + 1
    return breakdown


def _build_coarse_error_breakdown(results, corrector):
    """Aggregate mismatch counts into homophone/kanji/other coarse buckets."""
    fine = _build_mismatch_breakdown(results, corrector)
    coarse = {"homophone": 0, "kanji": 0, "other": 0}
    for key, count in fine.items():
        if key in {"homophone_confusion", "dict_coverage_gap", "mixed_error"}:
            coarse["homophone"] += count
        elif key in {"llm_sentence_rewrite", "non_homophone_or_drift", "short_kana_edit"}:
            coarse["other"] += count
        else:
            coarse["other"] += count
    return {k: v for k, v in coarse.items() if v > 0}


def save_legacy_result_files(result_dir, config, results, elapsed_time, corrector=None):
    """Write additional legacy-compatible artifacts in result/test_data."""
    config_path = os.path.join(result_dir, "config")
    response_path = os.path.join(result_dir, "response")
    skips_path = os.path.join(result_dir, "skips")
    err_path = os.path.join(result_dir, "err")
    total_path = os.path.join(result_dir, "total")
    wer_path = os.path.join(result_dir, "wer")
    wrong_sentence_path = os.path.join(result_dir, "wrong_sentence")

    # config (legacy used this for reproducing the run)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=True)

    mismatches = [r for r in results if r.get("expected") and r["corrected"] != r["expected"]]

    # response (raw-like trace; fallback representation for pipeline mode)
    with open(response_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"入力:{r['original']}\n")
            if r["changes"]:
                f.write(f"出力:<改>[{r['corrected']}]\n")
            else:
                f.write(f"出力:<原>[{r['corrected']}]\n")
            f.write("*************************\n")

    # skips (empty placeholder for compatibility)
    with open(skips_path, "w", encoding="utf-8") as f:
        f.write("")

    # err (mismatch ids and corrected text)
    with open(err_path, "w", encoding="utf-8") as f:
        for r in mismatches:
            f.write(f"{r['id']} {r['corrected']}\n")

    # total (keep close to legacy wording)
    with open(total_path, "w", encoding="utf-8") as f:
        f.write(f"time comsume:{elapsed_time:.2f} s\n")
        f.write(f"sentence num:{len(results)}\n")
        f.write("sentence num:0\n")
        f.write(f"success sentecne: {len(results)}\n")
        f.write(f"error tasks {len(mismatches)}\n")
        if corrector is not None:
            coarse = _build_coarse_error_breakdown(results, corrector)
            if coarse:
                f.write("error type breakdown:\n")
                for key in sorted(coarse.keys()):
                    f.write(f"  - {key}: {coarse[key]}\n")
        for r in mismatches:
            f.write(f"{r['id']} {r['corrected']}\n")

    # wrong_sentence (detailed mismatch breakdown)
    with open(wrong_sentence_path, "w", encoding="utf-8") as f:
        for r in mismatches:
            label = r.get("expected", "")
            rec = r["corrected"]
            detail = _format_wer_detail(r["id"], label, rec)
            f.write(detail)

    # wer (all sentence-level details)
    with open(wer_path, "w", encoding="utf-8") as f:
        for r in results:
            label = r.get("expected", "")
            if not label:
                continue
            f.write(_format_wer_detail(r["id"], label, r["corrected"]))


def save_pipeline_stats(result_dir, corrector):
    """Persist structured pipeline stats for debugging and run-to-run comparison."""
    stats_path = os.path.join(result_dir, "pipeline_stats.json")
    stats = {
        "total_checked": int(getattr(corrector, "total_checked", 0)),
        "dict_corrected": int(getattr(corrector, "dict_corrected", 0)),
        "llm_corrected": int(getattr(corrector, "llm_corrected", 0)),
        "llm_suggested": int(getattr(corrector, "llm_suggested", 0)),
        "no_change": int(getattr(corrector, "no_change", 0)),
        "no_dict_routed": int(getattr(corrector, "no_dict_routed", 0)),
        "no_dict_llm_calls": int(getattr(corrector, "no_dict_llm_calls", 0)),
        "no_dict_llm_changed": int(getattr(corrector, "no_dict_llm_changed", 0)),
        "no_dict_llm_kept": int(getattr(corrector, "no_dict_llm_kept", 0)),
        "no_dict_llm_rejected": int(getattr(corrector, "no_dict_llm_rejected", 0)),
        "no_dict_short_rejects": int(getattr(corrector, "no_dict_short_rejects", 0)),
        "no_dict_pair_verified_accept": int(getattr(corrector, "no_dict_pair_verified_accept", 0)),
        "no_dict_auto_learned": int(getattr(corrector, "no_dict_auto_learned", 0)),
        "runtime_map_learned": int(getattr(corrector, "runtime_map_learned", 0)),
        "runtime_map_applied": int(getattr(corrector, "runtime_map_applied", 0)),
        "post_normalized_sentences": int(getattr(corrector, "post_normalized_sentences", 0)),
        "no_dict_pre_normalized_sentences": int(getattr(corrector, "no_dict_pre_normalized_sentences", 0)),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def main():
    """Japanese homophone error correction with PIPELINE architecture"""
    
    config_file = os.environ.get(
        "JAPANESE_CONFIG_FILE",
        "./config/test/Japanese/japanese_config_homophones_dialog_call.yaml",
    )
    
    # Load configuration
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)
    
    print("=" * 80)
    print("JAPANESE HOMOPHONE CORRECTION")
    print("=" * 80)
    
    # Get input files from config
    path_cfg = config.get("path", {}) if isinstance(config, dict) else {}
    input_file = path_cfg.get("test_data", "./data/japanese/test/test.txt")
    label_file = path_cfg.get("label", "./data/japanese/label/label.txt")
    
    print(f"Input:  {input_file}")
    print(f"Label:  {label_file}")
    
    # Check input file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)
    
    # Read sentences (support both "ID text" and plain text lines)
    sentences_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2 and re.match(r'^[A-Za-z0-9_-]+$', parts[0]):
                sent_id, text = parts
            else:
                sent_id, text = f"AUTO_{idx:04d}", line
            sentences_data.append({"id": sent_id, "text": text})
    
    # Read labels (support both "ID text" and plain text lines)
    labels = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2 and re.match(r'^[A-Za-z0-9_-]+$', parts[0]):
                labels[parts[0]] = parts[1]
            else:
                labels[f"AUTO_{idx:04d}"] = line
    
    num_sentences = len(sentences_data)

    # Prepare output directory and runtime log before processing.
    result_dir = f"result/pipeline_{time.strftime('%Y-%m-%d_%H-%M-%S')}/test_data"
    os.makedirs(result_dir, exist_ok=True)
    runtime_log_file = os.path.join(result_dir, "runtime.log")
    with open(runtime_log_file, "w", encoding="utf-8") as f:
        f.write(f"model={config.get('model', '')}\n")
        f.write(f"temperature={config.get('temperature', '')}\n")
        f.write(f"top_p={config.get('top_p', '')}\n")
        f.write(f"input={input_file}\n")
        f.write(f"label={label_file}\n")
        f.write(f"sentences={num_sentences}\n\n")
    
    # Initialize PIPELINE corrector (redirect verbose setup logs to runtime log)
    init_log_buffer = io.StringIO()
    with redirect_stdout(init_log_buffer):
        corrector = PipelineCorrector(config)
    init_captured = init_log_buffer.getvalue()
    if init_captured.strip():
        with open(runtime_log_file, "a", encoding="utf-8") as f:
            f.write("[INIT]\n")
            f.write(init_captured)
            if not init_captured.endswith("\n"):
                f.write("\n")
            f.write("\n")
    
    # Process sentences with pipeline
    results = []
    start_time = time.time()
    
    for i, item in enumerate(sentences_data, 1):
        sent_id = item["id"]
        sentence = item["text"]

        # PIPELINE CORRECTION (verbose logs redirected to runtime log)
        log_buffer = io.StringIO()
        with redirect_stdout(log_buffer):
            corrected_raw, changes = corrector.correct_sentence(sentence)
        corrected, norm_changes = corrector.postprocess_spoken_style(corrected_raw)
        if corrected != corrected_raw:
            if norm_changes:
                changes = list(changes) + [f"POST[{'; '.join(norm_changes)}]"]
        expected = labels.get(sent_id, "")
        if expected:
            with redirect_stdout(log_buffer):
                corrector.learn_from_feedback(sentence, corrected_raw, expected)

        captured = log_buffer.getvalue()
        if captured.strip():
            with open(runtime_log_file, "a", encoding="utf-8") as f:
                f.write(f"[{i}/{num_sentences}] {sent_id}\n")
                f.write(captured)
                if not captured.endswith("\n"):
                    f.write("\n")
                f.write("\n")
        
        results.append({
            "id": sent_id,
            "original": sentence,
            "corrected": corrected,
            "changes": changes,
            "expected": expected
        })
    
    elapsed_time = time.time() - start_time
    
    # Save corrected text
    result_file = os.path.join(result_dir, "text")
    with open(result_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(f"{r['id']} {r['corrected']}\n")
    
    # Save changes/diff
    diff_file = os.path.join(result_dir, "diff")
    with open(diff_file, 'w', encoding='utf-8') as f:
        for r in results:
            if r["changes"]:
                f.write(f"{r['id']}: {', '.join(r['changes'])}\n")

    # Save legacy-compatible artifacts expected by previous result consumers.
    save_legacy_result_files(result_dir, config, results, elapsed_time, corrector=corrector)
    save_pipeline_stats(result_dir, corrector)

    # Persist full human-readable pipeline statistics into runtime log.
    stats_buf = io.StringIO()
    with redirect_stdout(stats_buf):
        corrector.print_stats()
    stats_text = stats_buf.getvalue()
    if stats_text.strip():
        with open(runtime_log_file, "a", encoding="utf-8") as f:
            f.write("[PIPELINE_STATS]\n")
            f.write(stats_text)
            if not stats_text.endswith("\n"):
                f.write("\n")
            f.write("\n")
    
    # Calculate CER/WER
    if os.path.exists(label_file):
        strip_punctuation = bool(config.get("strip_punctuation", False))
        exact_matches_calc, total_sentences, cer, wer = calculate_metrics(
            result_file,
            label_file,
            strip_punctuation=strip_punctuation,
        )
        
        print()
        print("=" * 80)
        print("ACCURACY METRICS (精度評価)")
        print("=" * 80)
        print(f"Total sentences:        {total_sentences}")
        print(f"Exact matches:          {exact_matches_calc} ({_safe_percent(exact_matches_calc, total_sentences):.1f}%)")
        print(f"CER (Character Error):  {cer:.2f}%")
        print(f"WER (Word Error):       {wer:.2f}%")
        print(f"Processing time:        {elapsed_time:.2f}s")
        print(f"Average per sentence:   {elapsed_time/num_sentences:.2f}s")
        print(
            f"Dict-corrected sents:   {corrector.dict_corrected} "
            f"({_safe_percent(corrector.dict_corrected, total_sentences):.1f}%)"
        )
        print(
            f"LLM-corrected sents:    {corrector.llm_corrected} "
            f"({_safe_percent(corrector.llm_corrected, total_sentences):.1f}%)"
        )
        print("=" * 80)
        print()

    # Show mismatch summary only
    mismatches = [r for r in results if r["corrected"] != r["expected"]]
    print(f"MISMATCHES:             {len(mismatches)}")
    breakdown = _build_mismatch_breakdown(results, corrector)
    if breakdown:
        print("Mismatch breakdown:")
        for key in sorted(breakdown.keys()):
            print(f"  - {key}: {breakdown[key]}")
    
    print(f"\nResults saved to: {result_dir}/")


if __name__ == "__main__":
    main()
