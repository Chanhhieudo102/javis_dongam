#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import time
import os
import json
import math
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from openai import OpenAI
from openai import APIError, APIConnectionError, APITimeoutError, RateLimitError
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    from tools.japanese_mecab_helper import JapaneseMorphAnalyzer
except Exception:
    class JapaneseMorphAnalyzer:
        def __init__(self, dicdir=""):
            self.available = False
            self.tagger = None

        def tokenize(self, text):
            return []

try:
    import MeCab
except Exception:
    MeCab = None

class PipelineCorrector:
    def __init__(self, config):
        self.model = config["model"]
        self.api_key = config["api_key"]
        self.base_url = config["base_url"]
        # Force a single model across all branches to keep behavior consistent.
        self.no_dict_model = self.model
        self.no_dict_api_key = config.get("no_dict_api_key", self.api_key)
        self.no_dict_base_url = config.get("no_dict_base_url", self.base_url)
        self.temperature = config.get("temperature", 0.0)
        self.top_p = config.get("top_p", 0.3)
        self.dict_confidence_threshold = config.get("dict_confidence_threshold", 4)
        self.max_api_retries = config.get("max_api_retries", 3)
        self.initial_backoff = config.get("initial_backoff", 1.0)
        self.backoff_multiplier = config.get("backoff_multiplier", 2.0)
        self.max_backoff = config.get("max_backoff", 30.0)
        self.reasoning_effort = config.get("reasoning_effort", None)
        self.use_groq_sdk = bool(config.get("use_groq_sdk", True))

        if self.use_groq_sdk and Groq is not None and "api.groq.com" in (self.base_url or ""):
            self.client = Groq(api_key=self.api_key)
            if self.no_dict_api_key == self.api_key and "api.groq.com" in (self.no_dict_base_url or ""):
                self.no_dict_client = self.client
            else:
                self.no_dict_client = Groq(api_key=self.no_dict_api_key)
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0)
            if self.no_dict_api_key == self.api_key and self.no_dict_base_url == self.base_url:
                self.no_dict_client = self.client
            else:
                self.no_dict_client = OpenAI(api_key=self.no_dict_api_key, base_url=self.no_dict_base_url, max_retries=0)
        
        self.auto_learn_dictionary = config.get("auto_learn_dictionary", True)
        self.no_dict_min_score = config.get("no_dict_min_score", 3)
        self.no_dict_relaxed_margin = config.get("no_dict_relaxed_margin", 2)
        self.no_dict_max_span_len = config.get("no_dict_max_span_len", 4)
        self.no_dict_max_len_delta = config.get("no_dict_max_len_delta", 2)
        self.protected_compounds = set(config.get("protected_compounds", []) or [])
        # Keep backward compatibility but default to full Japanese homophones
        # (kana/katakana/kanji), not kanji-only substitutions.
        self.no_dict_require_kanji = config.get("no_dict_require_kanji", False)
        self.no_dict_avoid_short_words = config.get("no_dict_avoid_short_words", True)
        self.no_dict_token_boundary_check = config.get("no_dict_token_boundary_check", True)
        self.auto_learn_min_len = config.get("auto_learn_min_len", 2)
        # Always learn directly into the main homophone dictionary.
        self.auto_learn_context_size = config.get("auto_learn_context_size", 2)
        # Force MeCab/POS filtering on for all runs.
        self.enable_pos_filter = True
        self.use_verified_runtime_map = config.get("use_verified_runtime_map", True)
        self.verified_runtime_min_hits = config.get("verified_runtime_min_hits", 1)
        self.no_dict_require_reading_match = config.get("no_dict_require_reading_match", True)
        self.no_dict_min_reading_similarity = float(config.get("no_dict_min_reading_similarity", 0.97))
        self.no_dict_adaptive_similarity_floor = float(config.get("no_dict_adaptive_similarity_floor", 0.85))
        self.no_dict_require_explicit_homophone = config.get("no_dict_require_explicit_homophone", True)
        self.no_dict_allow_oov_homophone = config.get("no_dict_allow_oov_homophone", True)
        self.no_dict_force_when_detector_ok = config.get("no_dict_force_when_detector_ok", False)
        self.no_dict_route_unresolved = config.get("no_dict_route_unresolved", True)
        self.enable_stage3_fallback = config.get("enable_stage3_fallback", True)
        self.disable_keep_choice = bool(config.get("disable_keep_choice", False))
        self.force_change_on_llm_uncertain = bool(config.get("force_change_on_llm_uncertain", False))
        # When enabled, unresolved (no-dict) cases are fully delegated to LLM:
        # LLM decides whether to change or keep, without extra no-dict gate rejection.
        self.no_dict_delegate_to_llm = config.get("no_dict_delegate_to_llm", False)
        self.no_dict_trust_llm_decision = config.get("no_dict_trust_llm_decision", False)
        self.no_dict_allow_sentence_level = config.get("no_dict_allow_sentence_level", True)
        self.no_dict_enable_span_guard = config.get("no_dict_enable_span_guard", False)
        self.no_dict_enable_pos_guard = config.get("no_dict_enable_pos_guard", False)
        self.disable_llm_on_rate_limit = config.get("disable_llm_on_rate_limit", True)
        self.no_dict_disable_hard_fixes = bool(config.get("no_dict_disable_hard_fixes", False))
        self.enable_no_dict_detector = config.get("enable_no_dict_detector", True)
        self.prioritize_pos_anomaly = config.get("prioritize_pos_anomaly", True)
        self.no_dict_auto_learn_from_llm = config.get("no_dict_auto_learn_from_llm", True)
        self.no_dict_auto_learn_min_hits = int(config.get("no_dict_auto_learn_min_hits", 2))
        self.no_dict_auto_learn_min_similarity = float(config.get("no_dict_auto_learn_min_similarity", 0.86))
        self.no_dict_auto_learn_source = config.get("no_dict_auto_learn_source", "auto_runtime")
        self.workflow_simple_mode = config.get("workflow_simple_mode", False)
        self.dict_recheck_margin = config.get("dict_recheck_margin", 2)
        self.llm_recheck_dict = config.get("llm_recheck_dict", False)
        self.llm_recheck_min_score = config.get("llm_recheck_min_score", 3)
        self.enable_embedding_recheck = config.get("enable_embedding_recheck", False)
        self.enable_llm_select = bool(config.get("enable_llm_select", False))
        self.llm_skip_tokens = set(config.get("llm_skip_tokens", ["はい", "えっと", "あの", "あ", "うーん"]) or [])
        self.skip_filler_in_dict_stage = bool(config.get("skip_filler_in_dict_stage", True))
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.embedding_recheck_max_candidates = int(config.get("embedding_recheck_max_candidates", 6))
        self.embedding_min_margin = float(config.get("embedding_min_margin", 0.05))
        self.log_pos_prunes = config.get("log_pos_prunes", True)
        self.pos_prune_log_path = config.get("pos_prune_log_path", "result/pos_prunes_homophone.jsonl")
        self.pos_filter_rules = config.get("pos_filter_rules", {}) or {}
        self.pos_allow = set(self.pos_filter_rules.get("allow", []) or [])
        self.pos_deny = set(self.pos_filter_rules.get("deny", []) or [])
        self.enable_collocation_score = config.get("enable_collocation_score", True)
        self.collocation_weight = float(config.get("collocation_weight", 3.5))
        self.collocation_window_size = int(config.get("collocation_window_size", 1))
        self.scoring_context_window = int(config.get("scoring_context_window", 5))
        # Optional signal only: default off for latency.
        self.enable_embedding_score_in_dict = config.get("enable_embedding_score_in_dict", False)
        self.embedding_score_weight = float(config.get("embedding_score_weight", 2.5))
        self.collocation_smoothing = float(config.get("collocation_smoothing", 0.5))
        self.collocation_min_count = int(config.get("collocation_min_count", 1))
        self.collocation_max_lines = int(config.get("collocation_max_lines", 200000))
        self.collocation_corpus_paths = config.get(
            "collocation_corpus_paths",
            [
                "data/japanese/label/test1.txt",
                "data/japanese/test/test1.txt",
            ],
        )
        verification_cfg = config.get("verification", {}) or {}
        self.verification_engine = verification_cfg.get("engine", "mecab")
        self.check_reading = verification_cfg.get("check_reading", True)
        self.strict_reading_ratio = float(verification_cfg.get("strict_reading_ratio", 1.0))
        self.no_dict_prompt_template = str(config.get("no_dict_prompt", "") or "").strip()
        self.no_dict_enable_pre_normalization = bool(config.get("no_dict_enable_pre_normalization", True))

        custom_pre_no_dict_rules = config.get("no_dict_pre_normalization_rules", []) or []
        raw_pre_no_dict_rules = []
        if isinstance(custom_pre_no_dict_rules, list):
            raw_pre_no_dict_rules.extend(custom_pre_no_dict_rules)
        self.no_dict_pre_normalization_rules = []
        for item in raw_pre_no_dict_rules:
            if not isinstance(item, dict):
                continue
            pattern = str(item.get("pattern", "") or "").strip()
            repl = str(item.get("repl", "") or "").strip()
            if not pattern or pattern == repl:
                continue
            self.no_dict_pre_normalization_rules.append((pattern, repl))
        self.enable_post_normalization = bool(config.get("enable_post_normalization", True))
        # Post-normalization map is now fully driven by config (no hardcoded defaults).
        post_norm_map = config.get("post_normalization_map", {}) or {}
        raw_post_map = {}
        if isinstance(post_norm_map, dict):
            raw_post_map = dict(post_norm_map)
        elif isinstance(post_norm_map, list):
            for item in post_norm_map:
                if not isinstance(item, dict):
                    continue
                src = str(item.get("src", "") or "").strip()
                dst = str(item.get("dst", "") or "").strip()
                if src and dst and src != dst:
                    raw_post_map[src] = dst
        # Build normalized_pairs from config (dict format only now)
        normalized_pairs = []
        for src, dst in dict(raw_post_map).items():
            src_s = str(src or "").strip()
            dst_s = str(dst or "").strip()
            if src_s and dst_s and src_s != dst_s:
                normalized_pairs.append((src_s, dst_s))
        self.post_normalization_pairs = sorted(normalized_pairs, key=lambda x: len(x[0]), reverse=True)
        self.no_dict_max_completion_tokens = config.get("max_completion_tokens", 120)
        self.llm_select_max_completion_tokens = config.get("llm_select_max_completion_tokens", 20)
        self.fast_llm_prompt = config.get("fast_llm_prompt", False)
        self.disable_dictionary_stage = bool(config.get("disable_dictionary_stage", False))
        self.mecab_dicdir = config.get("mecab_dicdir")
        self.homophones_file = config.get("homophones_file", "data/japanese/dictionary/homophones.json")
        self.dict_use_metadata_priority = config.get("dict_use_metadata_priority", True)
        self.dict_freq_weight = float(config.get("dict_freq_weight", 0.8))
        self.dict_source_weights = config.get(
            "dict_source_weights",
            {
                "human_verified": 3.0,
                "llm_verified": 2.0,
                "manual": 1.5,
                "auto_learned": 0.5,
            },
        )
        self.dict_boundary_override_sources = set(
            config.get(
                "dict_boundary_override_sources",
                ["human_verified", "llm_verified", "auto_wrong_sentence"],
            )
            or []
        )
        self.temporal_context_tokens = set(
            config.get(
                "temporal_context_tokens",
                [
                    "今日",
                    "明日",
                    "明後日",
                    "先週",
                    "来週",
                    "今週",
                    "先月",
                    "来月",
                    "以降",
                    "以前",
                    "まで",
                    "から",
                    "頃",
                    "時",
                    "日時",
                ],
            )
            or []
        )
        self.temporal_candidates = set(config.get("temporal_candidates", ["以降", "以前", "移行"]) or [])
        self.opinion_candidates = set(config.get("opinion_candidates", ["意向", "意見"]) or [])
        self.temporal_context_bonus = float(config.get("temporal_context_bonus", 3.0))
        self.temporal_context_penalty = float(config.get("temporal_context_penalty", 2.0))
        self.auto_learn_apply_on_verified_hits = config.get("auto_learn_apply_on_verified_hits", False)
        self.verified_runtime_path = config.get("verified_runtime_path", "result/runtime_verified_map_homophone.json")
        self.verified_runtime_ttl_hours = int(config.get("verified_runtime_ttl_hours", 168))
        self.verified_runtime_persist_min_hits = int(
            config.get("verified_runtime_persist_min_hits", self.verified_runtime_min_hits)
        )
        self.runtime_map_blacklist_path = config.get("runtime_map_blacklist_path", "result/runtime_blacklist_homophone.jsonl")
        self.blacklist_on_incorrect_feedback = bool(config.get("blacklist_on_incorrect_feedback", False))

        self._mecab_tagger = None
        self._mecab_yomi_tagger = None
        self._morph_analyzer = None
        self._candidate_pos_cache = {}
        self._reading_cache = {}
        self._morpheme_cache = {}
        self._collocation_counts = defaultdict(int)
        self._collocation_left_totals = defaultdict(int)
        self._collocation_vocab = set()
        self._collocation_total_pairs = 0
        self._verified_mapping_hits = {}
        self._verified_runtime_map = {}
        self._runtime_blacklist = set()
        self.homophones = {}
        self._reading_groups = {}
        self._candidate_rule_index = {}
        self._reading_candidate_map = {}
        self._reading_context_map = {}
        self._embedding_cache = {}
        self._embedding_available = True
        self._domain_profiles = self._compile_domain_profiles(config.get("domain_context_profiles", []))
        self._init_morph_analyzer()
        self._init_collocation_model()
        self.homophones = self._load_homophone_dictionary(self.homophones_file)
        self._build_reading_index()
        self._load_runtime_blacklist()
        self._load_verified_runtime_state()
        self._promote_runtime_map_to_dictionary()
        
        # Statistics
        self.total_checked = 0
        self.dict_corrected = 0
        self.llm_corrected = 0
        self.no_change = 0
        self.no_dict_routed = 0
        self.no_dict_llm_calls = 0
        self.no_dict_llm_changed = 0
        self.no_dict_llm_kept = 0
        self.no_dict_llm_rejected = 0
        self.no_dict_short_rejects = 0
        self.no_dict_pre_normalized_sentences = 0
        self.no_dict_pair_verified_accept = 0
        self.no_dict_auto_learned = 0
        self.pos_filtered_candidates = 0
        self.pos_prune_events = 0
        self.pos_guard_rejects = 0
        self.runtime_map_applied = 0
        self.runtime_map_learned = 0
        self.post_normalized_sentences = 0
        self.llm_recheck_calls = 0
        self.llm_recheck_dict_accept = 0
        self.llm_recheck_keep_override = 0
        self.llm_recheck_alt_accept = 0
        self.embedding_recheck_hints = 0
        self._last_used_llm = False
        self._llm_unavailable = False
        self._last_llm_pairs = []
        self._last_no_dict_pairs = []
        self._pending_no_dict_pairs = {}

    def _sanitize_dictionary_token(self, token):
        """Normalize candidate tokens from dictionary/runtime to avoid punctuation noise."""
        text = str(token or "").strip()
        if not text:
            return ""
        text = re.sub(r"^[\s、。,.!?！？；;:：]+|[\s、。,.!?！？；;:：]+$", "", text)
        if not text:
            return ""
        if re.search(r"[<>|\[\]{}]", text):
            return ""
        return text

    def _frequency_to_weight(self, frequency):
        """Normalize loader frequency labels into numeric score bonus."""
        if frequency is None:
            return 0.0
        if isinstance(frequency, (int, float)):
            return float(max(0.0, frequency))
        text = str(frequency).strip().lower()
        mapping = {
            "very_high": 8.0,
            "high": 5.0,
            "medium": 2.0,
            "low": 0.5,
        }
        if text in mapping:
            return mapping[text]
        try:
            return float(max(0.0, float(text)))
        except Exception:
            return 0.0

    def _frequency_to_int(self, frequency):
        """Map frequency labels to a compact integer used in learned-entry metadata."""
        weight = self._frequency_to_weight(frequency)
        if weight >= 8.0:
            return 10
        if weight >= 5.0:
            return 7
        if weight >= 2.0:
            return 4
        if weight > 0.0:
            return 2
        return 1

    def _is_protected_word(self, word):
        """No external vocabulary loader: keep simple default behavior."""
        return False

    def _get_vocab_frequency(self, word):
        """No external vocabulary loader: frequency is unavailable."""
        return None

    def _build_reading_index(self):
        """Build reading-to-candidate index to recover missed dictionary coverage."""
        self._reading_candidate_map = {}
        self._reading_context_map = {}
        for word, info in self.homophones.items():
            readings = info.get("readings", []) or []
            candidates = info.get("candidates", []) or []
            contexts = info.get("contexts", {}) or {}
            for reading in readings:
                if not reading:
                    continue
                bucket = self._reading_candidate_map.setdefault(reading, set())
                if word:
                    bucket.add(word)
                for candidate in candidates:
                    if candidate:
                        bucket.add(candidate)
                        key = (reading, candidate)
                        ctx_bucket = self._reading_context_map.setdefault(key, set())
                        for ctx in contexts.get(candidate, []) or []:
                            if ctx:
                                ctx_bucket.add(ctx)

    def _normalize_reading_key(self, reading):
        """Normalize dictionary reading key for robust matching."""
        cleaned = self._clean_reading_text(reading)
        if not cleaned:
            return ""
        return self._normalize_phonetic_text(cleaned)

    def _normalize_collocation_phrases(self, values):
        """Normalize collocation examples into compact semantic cores (N+particle+V)."""
        normalized = []
        pattern = re.compile(
            r"([\u30a0-\u30ff\u4e00-\u9fff々ヶA-Za-z0-9][\u3040-\u30ff\u4e00-\u9fff々ヶーA-Za-z0-9]{0,9})"
            r"(を|に|が|で)"
            r"([\u3040-\u30ff\u4e00-\u9fff々ヶーA-Za-z0-9]{1,10})"
        )
        for raw in values or []:
            phrase = re.sub(r"\s+", "", str(raw or "").strip())
            if not phrase:
                continue

            matches = list(pattern.finditer(phrase))
            if matches:
                noun, particle, verb = matches[-1].groups()
                if "の" in noun:
                    noun = noun.split("の")[-1]
                noun = re.sub(r"^[\u3040-\u309f]+", "", noun)
                verb = re.sub(r"(します|しました|して|した|する|できます|できる|でき|ます|ました|し|れる|られる)$", "", verb)
                if not verb:
                    _, _, fallback_verb = matches[-1].groups()
                    verb = fallback_verb
                core = f"{noun}{particle}{verb}"
                if core:
                    normalized.append(core)
                    continue

            if len(phrase) <= 12:
                normalized.append(phrase)

        return list(dict.fromkeys([p for p in normalized if p]))

    def _infer_pos_tags_for_word(self, word, current_tags):
        """Infer better POS tags from MeCab and overrides to avoid all-noun dictionaries."""
        token = (word or "").strip()
        tags = [str(t).strip() for t in (current_tags or []) if str(t).strip()]

        override_pos = {
            "すすめる": "動詞",
            "進める": "動詞",
            "勧める": "動詞",
            "いたす": "動詞",
            "致す": "動詞",
            "いたします": "動詞",
            "致します": "動詞",
            "板します": "動詞",
            "いたしました": "動詞",
            "致しました": "動詞",
            "いたしまた": "動詞",
            "はい": "感動詞",
            "ええ": "感動詞",
            "もし": "副詞",
            "とても": "副詞",
        }
        if token in override_pos:
            return [override_pos[token]]

        inferred = ""
        try:
            morphs = self._analyze_morphemes(token)
            for m in morphs:
                if (m.get("surface", "") or "").strip() != token:
                    continue
                pos = (m.get("pos", "") or "").strip()
                if pos:
                    inferred = pos
                    break
            if not inferred and len(morphs) == 1:
                inferred = (morphs[0].get("pos", "") or "").strip()
        except Exception:
            inferred = ""

        if inferred in {"名詞", "動詞", "形容詞", "形状詞", "感動詞", "副詞"}:
            if not tags or tags == ["名詞"]:
                return [inferred]
            return tags

        return tags if tags else ["名詞"]

    def _normalize_context_rules(self, context_rules):
        """Normalize context rule lists to stable list[str] values."""
        rules = context_rules if isinstance(context_rules, dict) else {}

        def _to_list(value):
            if isinstance(value, list):
                return [str(v).strip() for v in value if str(v).strip()]
            if isinstance(value, str):
                parts = re.split(r"[,、]\s*", value)
                return [p.strip() for p in parts if p.strip()]
            return []

        return {
            "require_any": _to_list(rules.get("require_any", [])),
            "require_collocation": self._normalize_collocation_phrases(_to_list(rules.get("require_collocation", []))),
            "exclude": _to_list(rules.get("exclude", [])),
        }

    def _build_from_reading_json_schema(self, payload):
        """Build internal homophone map from reading-centric JSON schema."""
        homophones = {}
        self._reading_groups = {}
        self._candidate_rule_index = {}

        for reading_key, group in (payload or {}).items():
            reading = self._normalize_reading_key(reading_key)
            if not reading:
                continue

            if not isinstance(group, dict):
                continue

            default_candidate = self._sanitize_dictionary_token(group.get("default_candidate", ""))
            raw_candidates = group.get("candidates", [])
            if not isinstance(raw_candidates, list):
                continue

            candidate_rules = {}
            words = []
            for item in raw_candidates:
                if not isinstance(item, dict):
                    continue
                word = self._sanitize_dictionary_token(item.get("word", ""))
                if not word:
                    continue
                pos_tags = item.get("pos_tags", [])
                if isinstance(pos_tags, str):
                    pos_tags = [pos_tags]
                pos_tags = [str(p).strip() for p in (pos_tags or []) if str(p).strip()]
                pos_tags = self._infer_pos_tags_for_word(word, pos_tags)
                context_rules = self._normalize_context_rules(item.get("context_rules", {}))
                weight = float(item.get("weight", 1.0) or 1.0)

                candidate_rules[word] = {
                    "word": word,
                    "pos_tags": pos_tags,
                    "context_rules": context_rules,
                    "weight": weight,
                }
                self._candidate_rule_index[(reading, word)] = candidate_rules[word]
                words.append(word)

            if not words:
                continue

            if default_candidate and default_candidate not in words:
                default_candidate = ""

            self._reading_groups[reading] = {
                "default_candidate": default_candidate,
                "words": words,
                "candidate_rules": candidate_rules,
            }

            for source_word in words:
                others = [w for w in words if w != source_word]
                if source_word not in homophones:
                    homophones[source_word] = {
                        "readings": [reading],
                        "candidates": [],
                        "contexts": {},
                        "metadata": {},
                    }
                elif reading not in homophones[source_word]["readings"]:
                    homophones[source_word]["readings"].append(reading)

                for candidate in others:
                    if candidate not in homophones[source_word]["candidates"]:
                        homophones[source_word]["candidates"].append(candidate)

                    rule = candidate_rules.get(candidate, {})
                    rule_ctx = rule.get("context_rules", {}) if isinstance(rule, dict) else {}
                    require_any = rule_ctx.get("require_any", []) if isinstance(rule_ctx, dict) else []
                    require_collocation = rule_ctx.get("require_collocation", []) if isinstance(rule_ctx, dict) else []
                    merged_contexts = list(dict.fromkeys((require_any or []) + (require_collocation or [])))
                    if merged_contexts:
                        homophones[source_word]["contexts"][candidate] = merged_contexts

                    freq = max(1, int(round(float(rule.get("weight", 1.0) or 1.0) * 3)))
                    homophones[source_word].setdefault("metadata", {})[candidate] = {
                        "freq": freq,
                        "source": "reading_json",
                        "weight": float(rule.get("weight", 1.0) or 1.0),
                        "context_rules": rule_ctx,
                        "pos_tags": list(rule.get("pos_tags", []) or []),
                        "default_candidate": bool(default_candidate and candidate == default_candidate),
                    }

        return homophones

    def _load_runtime_blacklist(self):
        """Load bad learned mappings to avoid re-applying known wrong pairs."""
        self._runtime_blacklist = set()
        path = self.runtime_map_blacklist_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                        wrong = (payload.get("wrong_word") or payload.get("wrong") or "").strip()
                        correct = (payload.get("correct_word") or payload.get("correct") or "").strip()
                        if wrong and correct:
                            self._runtime_blacklist.add((wrong, correct))
                    except Exception:
                        continue
        except Exception:
            pass

    def _is_blacklisted_mapping(self, wrong_word, correct_word):
        """Return True if mapping is listed in blacklist file."""
        return (wrong_word, correct_word) in self._runtime_blacklist

    def _append_runtime_blacklist(
        self,
        wrong_word,
        correct_word,
        reason="feedback_incorrect",
        context_original="",
        context_corrected="",
        context_expected="",
    ):
        """Append one bad mapping to blacklist file and in-memory set."""
        wrong = (wrong_word or "").strip()
        correct = (correct_word or "").strip()
        if not wrong or not correct or wrong == correct:
            return
        if self._is_blacklisted_mapping(wrong, correct):
            return

        self._runtime_blacklist.add((wrong, correct))
        if not self.runtime_map_blacklist_path:
            return

        try:
            os.makedirs(os.path.dirname(self.runtime_map_blacklist_path), exist_ok=True)
            payload = {
                "timestamp": int(time.time()),
                "wrong_word": wrong,
                "correct_word": correct,
                "reason": reason,
            }
            if context_original:
                payload["context_original"] = context_original
            if context_corrected:
                payload["context_corrected"] = context_corrected
            if context_expected:
                payload["context_expected"] = context_expected
            with open(self.runtime_map_blacklist_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _drop_runtime_mapping(self, wrong_word, correct_word=""):
        """Remove a mapping from verified runtime caches and persist state."""
        wrong = (wrong_word or "").strip()
        correct = (correct_word or "").strip()
        if not wrong:
            return

        if correct:
            self._verified_mapping_hits.pop((wrong, correct), None)
        else:
            for key in list(self._verified_mapping_hits.keys()):
                if key[0] == wrong:
                    self._verified_mapping_hits.pop(key, None)

        if wrong in self._verified_runtime_map:
            if not correct or self._verified_runtime_map.get(wrong) == correct:
                self._verified_runtime_map.pop(wrong, None)

        self._persist_verified_runtime_state()

    def _load_verified_runtime_state(self):
        """Load persisted verified runtime map and drop expired entries by TTL."""
        path = self.verified_runtime_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            now = int(time.time())
            ttl_sec = max(0, self.verified_runtime_ttl_hours) * 3600
            entries = payload.get("entries", {}) if isinstance(payload, dict) else {}
            for wrong, info in entries.items():
                if not isinstance(info, dict):
                    continue
                correct = (info.get("correct") or "").strip()
                hits = int(info.get("hits", 0) or 0)
                updated_at = int(info.get("updated_at", 0) or 0)
                if not wrong or not correct:
                    continue
                if self._is_blacklisted_mapping(wrong, correct):
                    continue
                if ttl_sec > 0 and updated_at > 0 and (now - updated_at) > ttl_sec:
                    continue
                self._verified_mapping_hits[(wrong, correct)] = max(self._verified_mapping_hits.get((wrong, correct), 0), hits)
                self._verified_runtime_map[wrong] = correct
        except Exception:
            pass

    def _persist_verified_runtime_state(self):
        """Persist verified runtime hits/map for reuse across runs."""
        if not self.verified_runtime_path:
            return
        try:
            os.makedirs(os.path.dirname(self.verified_runtime_path), exist_ok=True)
            entries = {}
            now = int(time.time())
            for (wrong, correct), hits in self._verified_mapping_hits.items():
                if hits < self.verified_runtime_persist_min_hits:
                    continue
                if self._is_blacklisted_mapping(wrong, correct):
                    continue
                entries[wrong] = {
                    "correct": correct,
                    "hits": int(hits),
                    "updated_at": now,
                }
            payload = {
                "schema": "runtime_verified_map_v1",
                "generated_at": now,
                "ttl_hours": self.verified_runtime_ttl_hours,
                "entries": entries,
            }
            with open(self.verified_runtime_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _promote_runtime_map_to_dictionary(self):
        """Promote persisted runtime verified mappings into the main JSON dictionary."""
        if not self.auto_learn_dictionary:
            return
        if not self._verified_runtime_map:
            return
        for wrong_word, correct_word in list(self._verified_runtime_map.items()):
            if not wrong_word or not correct_word or wrong_word == correct_word:
                continue
            hits = int(self._verified_mapping_hits.get((wrong_word, correct_word), 1) or 1)
            try:
                self._add_learned_homophone_entry(
                    wrong_word,
                    correct_word,
                    context_hint="runtime_verified",
                    source="llm_verified",
                    freq=hits,
                )
            except Exception:
                continue

    def _log_pos_prune(self, sentence, word, candidate, reason, source_pos=None, candidate_pos=None):
        """Persist detailed POS prune events for later rule tuning."""
        if not self.log_pos_prunes:
            return
        try:
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "sentence": sentence,
                "word": word,
                "candidate": candidate,
                "reason": reason,
                "source_pos": sorted(list(source_pos or [])),
                "candidate_pos": sorted(list(candidate_pos or [])),
            }
            os.makedirs(os.path.dirname(self.pos_prune_log_path), exist_ok=True)
            with open(self.pos_prune_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.pos_prune_events += 1
        except Exception:
            pass

    def _is_token_boundary_match(self, sentence, start, length):
        """Validate replacement span aligns with token boundaries."""
        if not self.no_dict_token_boundary_check:
            return True
        tokens = self._analyze_morphemes(sentence)
        if not tokens:
            return True
        end = start + length
        has_start = any(t["start"] == start for t in tokens)
        has_end = any(t["end"] == end for t in tokens)
        return has_start and has_end

    def _is_inside_protected_compound(self, sentence, start, length):
        """Return True when replacement span is contained in a protected lexical compound."""
        if not sentence or not self.protected_compounds:
            return False
        end = start + max(0, length)
        if end <= start:
            return False
        for term in self.protected_compounds:
            if not term:
                continue
            search_from = 0
            while True:
                idx = sentence.find(term, search_from)
                if idx < 0:
                    break
                term_end = idx + len(term)
                if start >= idx and end <= term_end:
                    return True
                search_from = idx + 1
        return False

    def _llm_recheck_dict_replacement(self, sentence, word, dict_choice, candidates):
        """Validate low-confidence dictionary choice before applying it."""
        self.llm_recheck_calls += 1
        candidate_text = ", ".join([c for c in candidates if c])
        embedding_scores = self._embedding_delta_scores(sentence, word, candidates, client=self.client)
        embedding_hint = ""
        if embedding_scores:
            ranked = sorted(embedding_scores.items(), key=lambda x: x[1], reverse=True)
            top_lines = [f"- {cand}: {score:+.3f}" for cand, score in ranked[:3]]
            leader = ranked[0][0]
            leader_margin = ranked[0][1] - float(embedding_scores.get(dict_choice, ranked[0][1]))
            preferred = ""
            if leader != dict_choice and leader_margin >= self.embedding_min_margin:
                preferred = f"\n補足: Embedding上位は {leader}（辞書候補との差分 {leader_margin:+.3f}）"
            embedding_hint = "\nEmbedding差分（高いほど文脈適合）:\n" + "\n".join(top_lines) + preferred + "\n"
        if self.disable_keep_choice:
            prompt = f"""【同音異義語修正・再判定】

文: {sentence}
対象語: {word}
辞書候補: {dict_choice}
選択肢: {candidate_text}
{embedding_hint}

【ルール】
1. 必ず変更を選ぶ（KEEP禁止）
2. 辞書候補が妥当なら ACTION: DICT
3. 選択肢内で辞書候補より良い語が明確なら ACTION: ALT | WORD: <候補>
4. WORDは必ず選択肢から選ぶ
5. 説明は禁止
"""
        else:
            prompt = f"""【同音異義語修正・再判定】

文: {sentence}
対象語: {word}
辞書候補: {dict_choice}
選択肢: {candidate_text}, KEEP_ORIGINAL
{embedding_hint}

【ルール】
1. 辞書候補が妥当なら ACTION: DICT
2. 辞書候補が不適切なら ACTION: KEEP
3. 選択肢内で辞書候補より良い語が明確なら ACTION: ALT | WORD: <候補>
4. 説明は禁止
"""
        try:
            response = self._call_llm_with_retry(prompt, max_tokens=30)
            raw = self._strip_reasoning_blocks(response.choices[0].message.content)
            upper = raw.upper()
            if "ACTION:" not in upper:
                return "DICT", dict_choice
            if "ACTION: KEEP" in upper and not self.disable_keep_choice:
                return "KEEP", word
            if "ACTION: DICT" in upper:
                return "DICT", dict_choice
            if "ACTION: ALT" in upper:
                alt_match = re.search(r"WORD\s*:\s*([^|\n]+)", raw, flags=re.IGNORECASE)
                if not alt_match:
                    return "DICT", dict_choice
                alt = alt_match.group(1).strip().replace("<", "").replace(">", "")
                if alt and alt in candidates and alt != word:
                    return "ALT", alt
            return "DICT", dict_choice
        except Exception:
            return "DICT", dict_choice

    def _load_homophone_dictionary(self, filepath):
        """Load homophone dictionary from reading-centric JSON schema."""
        self._reading_groups = {}
        self._candidate_rule_index = {}
        homophones = {}
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, using empty dictionary")
            return homophones

        if not str(filepath).lower().endswith(".json"):
            print(f"Warning: only JSON homophone dictionary is supported now ({filepath})")
            return homophones

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                homophones = self._build_from_reading_json_schema(payload)
                print(f"Loaded {len(self._reading_groups)} reading groups from {filepath}")
                return homophones
        except Exception as e:
            print(f"Warning: failed to parse reading JSON dictionary ({filepath}): {e}")

        return homophones

    def _find_homophones(self, sentence):
        """Find potential homophone errors in sentence."""
        results = []
        seen = set()
        for token in self._analyze_morphemes(sentence):
            word = (token.get("surface") or "").strip()
            if not word:
                continue
            if not re.search(r"[\u3040-\u30ff\u4e00-\u9fff々ヶー]", word):
                continue

            reading = self._normalize_reading_key(token.get("reading", ""))
            if not reading:
                reading = self._normalize_reading_key(self._reading_for_text(word))

            candidates = []
            candidate_rules = {}
            contexts = {}

            group = self._reading_groups.get(reading) if reading else None
            if group:
                all_words = [w for w in group.get("words", []) if w]
                if not all_words:
                    continue

                candidates = [w for w in all_words if w != word]
                if not candidates:
                    continue

                default_candidate = group.get("default_candidate", "")
                if default_candidate and default_candidate in candidates:
                    candidates = [default_candidate] + [c for c in candidates if c != default_candidate]

                for candidate in candidates:
                    rule = dict(group.get("candidate_rules", {}).get(candidate, {}) or {})
                    candidate_rules[candidate] = rule
                    rule_ctx = rule.get("context_rules", {}) if isinstance(rule, dict) else {}
                    require_any = rule_ctx.get("require_any", []) if isinstance(rule_ctx, dict) else []
                    require_collocation = rule_ctx.get("require_collocation", []) if isinstance(rule_ctx, dict) else []
                    contexts[candidate] = list(dict.fromkeys((require_any or []) + (require_collocation or [])))
            else:
                # Fallback for analyzers that return non-phonetic readings (e.g., surface kanji).
                info = self.homophones.get(word, {}) or {}
                candidates = [c for c in info.get("candidates", []) if c and c != word]
                if not candidates:
                    continue
                metadata = info.get("metadata", {}) if isinstance(info, dict) else {}
                raw_contexts = info.get("contexts", {}) if isinstance(info, dict) else {}
                contexts = {k: list(v or []) for k, v in (raw_contexts or {}).items()}
                for candidate in candidates:
                    meta = metadata.get(candidate, {}) if isinstance(metadata, dict) else {}
                    rule = {
                        "pos_tags": list(meta.get("pos_tags", []) or []),
                        "context_rules": dict(meta.get("context_rules", {}) or {}),
                        "weight": float(meta.get("weight", 1.0) or 1.0),
                    }
                    candidate_rules[candidate] = rule
                    rule_ctx = rule.get("context_rules", {}) if isinstance(rule, dict) else {}
                    require_any = rule_ctx.get("require_any", []) if isinstance(rule_ctx, dict) else []
                    require_collocation = rule_ctx.get("require_collocation", []) if isinstance(rule_ctx, dict) else []
                    if candidate not in contexts:
                        contexts[candidate] = list(dict.fromkeys((require_any or []) + (require_collocation or [])))

            start = int(token.get("start", sentence.find(word)))
            key = (start, word)
            if key in seen:
                continue
            seen.add(key)

            results.append(
                {
                    "word": word,
                    "position": start,
                    "candidates": candidates,
                    "contexts": contexts,
                    "candidate_rules": candidate_rules,
                    "reading": reading,
                }
            )

        # Fallback: MeCab can split unknown compounds and miss newly learned words.
        # Scan dictionary surfaces directly so learned entries are reusable next runs.
        for source_word in sorted(self.homophones.keys(), key=lambda x: len(x), reverse=True):
            token = (source_word or "").strip()
            if len(token) < 2:
                continue
            if not re.search(r"[\u3040-\u30ff\u4e00-\u9fff々ヶー]", token):
                continue

            start = sentence.find(token)
            if start < 0:
                continue
            key = (start, token)
            if key in seen:
                continue

            info = self.homophones.get(token, {}) or {}
            candidates = [c for c in info.get("candidates", []) if c and c != token]
            if not candidates:
                continue

            metadata = info.get("metadata", {}) if isinstance(info, dict) else {}
            raw_contexts = info.get("contexts", {}) if isinstance(info, dict) else {}
            contexts = {k: list(v or []) for k, v in (raw_contexts or {}).items()}
            candidate_rules = {}
            for candidate in candidates:
                meta = metadata.get(candidate, {}) if isinstance(metadata, dict) else {}
                rule = {
                    "pos_tags": list(meta.get("pos_tags", []) or []),
                    "context_rules": dict(meta.get("context_rules", {}) or {}),
                    "weight": float(meta.get("weight", 1.0) or 1.0),
                }
                candidate_rules[candidate] = rule
                rule_ctx = rule.get("context_rules", {}) if isinstance(rule, dict) else {}
                require_any = rule_ctx.get("require_any", []) if isinstance(rule_ctx, dict) else []
                require_collocation = rule_ctx.get("require_collocation", []) if isinstance(rule_ctx, dict) else []
                if candidate not in contexts:
                    contexts[candidate] = list(dict.fromkeys((require_any or []) + (require_collocation or [])))

            seen.add(key)
            results.append(
                {
                    "word": token,
                    "position": start,
                    "candidates": candidates,
                    "contexts": contexts,
                    "candidate_rules": candidate_rules,
                    "reading": "",
                }
            )
        return results

    def _tokenize_context_keywords(self, values):
        """Normalize context keywords into lexical tokens."""
        tokens = []
        for value in values or []:
            if not value:
                continue
            parts = re.findall(r"[\u3040-\u30ff\u4e00-\u9fff々ヶA-Za-z0-9]{1,}", str(value))
            tokens.extend(parts)
        return tokens

    def _tfidf_cosine_similarity(self, source_tokens, target_tokens):
        """Compute lightweight TF-IDF cosine similarity between two token lists."""
        if not source_tokens or not target_tokens:
            return 0.0

        src = Counter(source_tokens)
        tgt = Counter(target_tokens)
        vocab = list(set(src.keys()) | set(tgt.keys()))
        if not vocab:
            return 0.0

        def _idf(term):
            df = int(term in src) + int(term in tgt)
            return math.log((2.0 + 1.0) / (df + 1.0)) + 1.0

        dot = 0.0
        src_norm = 0.0
        tgt_norm = 0.0
        for term in vocab:
            idf = _idf(term)
            s = src.get(term, 0) * idf
            t = tgt.get(term, 0) * idf
            dot += s * t
            src_norm += s * s
            tgt_norm += t * t

        if src_norm <= 0.0 or tgt_norm <= 0.0:
            return 0.0
        return dot / math.sqrt(src_norm * tgt_norm)

    def _cosine_similarity(self, a, b):
        """Compute cosine similarity for two dense vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(a, b):
            dot += x * y
            norm_a += x * x
            norm_b += y * y
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    def _get_text_embedding(self, text, client=None):
        """Fetch embedding vector with cache and graceful disable on unsupported providers."""
        if not self.enable_embedding_recheck:
            return None
        normalized = (text or "").strip()
        if not normalized:
            return None
        if normalized in self._embedding_cache:
            return self._embedding_cache.get(normalized)

        if not self._embedding_available:
            return None

        use_client = client or self.client
        try:
            response = use_client.embeddings.create(
                model=self.embedding_model,
                input=[normalized],
            )
            vector = response.data[0].embedding if getattr(response, "data", None) else None
            if vector:
                self._embedding_cache[normalized] = vector
            return vector
        except Exception:
            self._embedding_available = False
            return None

    def _embedding_delta_scores(self, sentence, word, candidates, client=None):
        """Return sim(context,candidate)-sim(context,source) for candidate ranking hints."""
        if not self.enable_embedding_recheck:
            return {}
        valid_candidates = [c for c in (candidates or []) if c and c != word]
        if not valid_candidates:
            return {}
        valid_candidates = valid_candidates[: max(1, self.embedding_recheck_max_candidates)]

        left_tokens, right_tokens, _, _ = self._extract_scoring_context_window(sentence, word, window_size=3)
        context_text = " ".join([tok for tok in (left_tokens + right_tokens) if tok]).strip()
        if not context_text:
            return {}
        source_text = " ".join([*(left_tokens or []), word, *(right_tokens or [])]).strip()

        context_emb = self._get_text_embedding(context_text, client=client)
        source_emb = self._get_text_embedding(source_text, client=client)
        if not context_emb or not source_emb:
            return {}

        base_sim = self._cosine_similarity(context_emb, source_emb)
        scores = {}
        for candidate in valid_candidates:
            cand_text = " ".join([*(left_tokens or []), candidate, *(right_tokens or [])]).strip()
            cand_emb = self._get_text_embedding(cand_text, client=client)
            if not cand_emb:
                continue
            scores[candidate] = self._cosine_similarity(context_emb, cand_emb) - base_sim

        if scores:
            self.embedding_recheck_hints += 1
        return scores

    def _extract_scoring_context_window(self, sentence, target, window_size=3):
        """Extract left/right token windows around target for candidate scoring."""
        if not sentence or not target:
            return [], [], "", ""

        tokens = self._analyze_morphemes(sentence)
        if not tokens:
            fallback = re.findall(r"[\u3040-\u30ff\u4e00-\u9fff々ヶA-Za-z0-9]+", sentence or "")
            token_spans = []
            cursor = 0
            for token in (fallback if fallback else list(sentence or "")):
                start = sentence.find(token, cursor)
                if start < 0:
                    start = cursor
                end = start + len(token)
                token_spans.append((token, start, end))
                cursor = end
            target_start = sentence.find(target)
            if target_start < 0:
                return [], [], "", ""
            target_end = target_start + len(target)
            center_idx = None
            for idx, (_, start, end) in enumerate(token_spans):
                if start < target_end and end > target_start:
                    center_idx = idx
                    break
            if center_idx is None:
                return [], [], sentence[max(0, target_start - 8):target_start], sentence[target_end:target_end + 8]
            left_tokens = [t[0] for t in token_spans[max(0, center_idx - window_size):center_idx]]
            right_tokens = [t[0] for t in token_spans[center_idx + 1:center_idx + 1 + window_size]]
            return left_tokens, right_tokens, "".join(left_tokens), "".join(right_tokens)

        target_start = sentence.find(target)
        if target_start < 0:
            return [], [], "", ""
        target_end = target_start + len(target)
        center_idx = None
        for idx, token in enumerate(tokens):
            if token["start"] < target_end and token["end"] > target_start:
                center_idx = idx
                break
        if center_idx is None:
            return [], [], sentence[max(0, target_start - 8):target_start], sentence[target_end:target_end + 8]

        left_slice = tokens[max(0, center_idx - window_size):center_idx]
        right_slice = tokens[center_idx + 1:center_idx + 1 + window_size]
        left_tokens = [t.get("surface", "") for t in left_slice if t.get("surface")]
        right_tokens = [t.get("surface", "") for t in right_slice if t.get("surface")]
        return left_tokens, right_tokens, "".join(left_tokens), "".join(right_tokens)

    def _iter_collocation_corpus_lines(self):
        """Yield plain-text lines from configured collocation corpus files."""
        line_count = 0
        for path in self.collocation_corpus_paths or []:
            if not path or not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for raw in f:
                        text = (raw or "").strip()
                        if not text:
                            continue
                        yield text
                        line_count += 1
                        if self.collocation_max_lines > 0 and line_count >= self.collocation_max_lines:
                            return
            except Exception:
                continue

    def _init_collocation_model(self):
        """Build a lightweight bigram table for collocation scoring."""
        if not self.enable_collocation_score:
            return

        for line in self._iter_collocation_corpus_lines():
            tokens = self.tokenize(line)
            if len(tokens) < 2:
                continue
            for i in range(len(tokens) - 1):
                left = (tokens[i] or "").strip()
                right = (tokens[i + 1] or "").strip()
                if not left or not right:
                    continue
                key = (left, right)
                self._collocation_counts[key] += 1
                self._collocation_left_totals[left] += 1
                self._collocation_vocab.add(right)
                self._collocation_total_pairs += 1

    def _bigram_log_prob(self, left_token, right_token):
        """Return smoothed log-probability for P(right | left)."""
        left = (left_token or "").strip()
        right = (right_token or "").strip()
        if not left or not right:
            return 0.0

        left_total = self._collocation_left_totals.get(left, 0)
        pair_count = self._collocation_counts.get((left, right), 0)
        vocab_size = max(1, len(self._collocation_vocab))
        smooth = max(1e-6, float(self.collocation_smoothing))
        num = pair_count + smooth
        den = left_total + smooth * vocab_size
        if den <= 0.0:
            return 0.0
        return math.log(num / den)

    def _collocation_score(self, source_word, candidate, left_tokens, right_tokens):
        """Score candidate by left/right bigram compatibility against source word."""
        if not self.enable_collocation_score:
            return 0.0
        if self._collocation_total_pairs <= 0:
            return 0.0

        delta = 0.0

        window = max(1, int(self.collocation_window_size))

        for offset in range(1, min(window, len(left_tokens)) + 1):
            left = left_tokens[-offset]
            cand_pair_count = self._collocation_counts.get((left, candidate), 0)
            src_pair_count = self._collocation_counts.get((left, source_word), 0)
            if cand_pair_count >= self.collocation_min_count or src_pair_count >= self.collocation_min_count:
                delta += self._bigram_log_prob(left, candidate) - self._bigram_log_prob(left, source_word)

        for offset in range(0, min(window, len(right_tokens))):
            right = right_tokens[offset]
            cand_pair_count = self._collocation_counts.get((candidate, right), 0)
            src_pair_count = self._collocation_counts.get((source_word, right), 0)
            if cand_pair_count >= self.collocation_min_count or src_pair_count >= self.collocation_min_count:
                delta += self._bigram_log_prob(candidate, right) - self._bigram_log_prob(source_word, right)

        # Clamp to prevent extreme values on sparse corpora.
        return max(-4.0, min(4.0, delta))

    def _has_context_token(self, sentence, token_set):
        """Return True if any context token exists in sentence."""
        text = sentence or ""
        for token in token_set or []:
            if token and token in text:
                return True
        return False

    def _compile_domain_profiles(self, profiles):
        """Normalize domain context profiles for world-logic scoring."""
        normalized = []
        for item in profiles or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip() or "default"
            context_tokens = [str(t).strip() for t in (item.get("context_tokens", []) or []) if str(t).strip()]
            unlikely_tokens = [str(t).strip() for t in (item.get("unlikely_tokens", []) or []) if str(t).strip()]
            preferred_tokens = [str(t).strip() for t in (item.get("preferred_tokens", []) or []) if str(t).strip()]
            unlikely_penalty = float(item.get("unlikely_penalty", 5.0) or 5.0)
            preferred_bonus = float(item.get("preferred_bonus", 1.0) or 1.0)
            if not context_tokens:
                continue
            normalized.append(
                {
                    "name": name,
                    "context_tokens": context_tokens,
                    "unlikely_tokens": unlikely_tokens,
                    "preferred_tokens": preferred_tokens,
                    "unlikely_penalty": unlikely_penalty,
                    "preferred_bonus": preferred_bonus,
                }
            )
        return normalized

    def _active_domain_profiles(self, sentence):
        """Return profiles activated by sentence-level context tokens."""
        text = sentence or ""
        active = []
        for profile in self._domain_profiles:
            context_tokens = profile.get("context_tokens", [])
            if any(tok and tok in text for tok in context_tokens):
                active.append(profile)
        return active

    def _world_logic_adjustment(self, sentence, candidate):
        """Penalty/bonus from domain world-logic profiles."""
        score = 0.0
        token = (candidate or "").strip()
        if not token:
            return score
        for profile in self._active_domain_profiles(sentence):
            unlikely = set(profile.get("unlikely_tokens", []) or [])
            preferred = set(profile.get("preferred_tokens", []) or [])
            if token in unlikely:
                score -= float(profile.get("unlikely_penalty", 5.0) or 5.0)
            if token in preferred:
                score += float(profile.get("preferred_bonus", 1.0) or 1.0)
        return score

    def _get_best_candidate(self, word, sentence, candidates, contexts):
        """Select best candidate using context, TF-IDF and reading compatibility."""
        if not candidates:
            return None, 0

        scores = {}
        left_tokens, right_tokens, left_text, right_text = self._extract_scoring_context_window(
            sentence,
            word,
            window_size=max(1, self.scoring_context_window),
        )
        context_window_tokens = left_tokens + right_tokens
        embedding_deltas = {}
        if self.enable_embedding_score_in_dict:
            embedding_deltas = self._embedding_delta_scores(sentence, word, candidates, client=self.client)
        temporal_context_active = self._has_context_token(sentence, self.temporal_context_tokens)
        has_temporal_option = any(c in self.temporal_candidates for c in candidates)
        has_opinion_option = any(c in self.opinion_candidates for c in candidates)
        temporal_disambiguation = temporal_context_active and has_temporal_option and has_opinion_option

        for candidate in candidates:
            score = 0.0
            candidate_keywords = self._tokenize_context_keywords(contexts.get(candidate, []))
            candidate_meta = (self.homophones.get(word, {}) or {}).get("metadata", {}).get(candidate, {})
            candidate_rules = candidate_meta.get("context_rules", {}) if isinstance(candidate_meta, dict) else {}
            require_any = candidate_rules.get("require_any", []) if isinstance(candidate_rules, dict) else []
            require_collocation = candidate_rules.get("require_collocation", []) if isinstance(candidate_rules, dict) else []
            exclude_tokens = candidate_rules.get("exclude", []) if isinstance(candidate_rules, dict) else []

            if candidate in contexts and contexts[candidate]:
                for context_word in contexts[candidate]:
                    if context_word in sentence:
                        word_pos = sentence.find(word)
                        context_pos = sentence.find(context_word)
                        if word_pos >= 0 and context_pos >= 0:
                            distance = abs(word_pos - context_pos)
                            if distance <= 3:
                                score += 5
                            elif distance <= 6:
                                score += 3
                            elif distance <= 10:
                                score += 2
                            else:
                                score += 1
                        if context_word in left_text or context_word in right_text:
                            score += 2

            # New schema rule score: require_any / require_collocation / exclude.
            if require_any:
                if any(tok and tok in sentence for tok in require_any):
                    score += 3
                else:
                    score -= 2

            if require_collocation:
                hit_count = 0
                for phrase in require_collocation:
                    if phrase and phrase in sentence:
                        hit_count += 1
                if hit_count > 0:
                    score += min(4, hit_count * 2)
                else:
                    score -= 2

            if exclude_tokens and any(tok and tok in sentence for tok in exclude_tokens):
                score -= 4

            score += self._tfidf_cosine_similarity(context_window_tokens, candidate_keywords) * 8.0

            collocation_delta = self._collocation_score(word, candidate, left_tokens, right_tokens)
            score += collocation_delta * self.collocation_weight

            if candidate in embedding_deltas:
                score += float(embedding_deltas.get(candidate, 0.0) or 0.0) * self.embedding_score_weight

            reading_sim = self._reading_similarity(self._reading_for_text(word), self._reading_for_text(candidate))
            if reading_sim < 0.55:
                score -= 3
            elif reading_sim >= 0.90:
                score += 2
            elif reading_sim >= 0.75:
                score += 1

            if left_tokens and f"{left_tokens[-1]}{candidate}" in sentence:
                score += 2
            if right_tokens and f"{candidate}{right_tokens[0]}" in sentence:
                score += 2

            if candidate in sentence and candidate != word:
                score += 6

            if self.dict_use_metadata_priority:
                freq = max(1, int(candidate_meta.get("freq", 1) or 1))
                source = str(candidate_meta.get("source", "manual") or "manual")
                score += math.log(1.0 + freq) * self.dict_freq_weight
                score += float(self.dict_source_weights.get(source, 0.0))
                score += float(candidate_meta.get("weight", 1.0) or 1.0)

            vocab_freq = self._get_vocab_frequency(candidate)
            if vocab_freq is not None:
                score += self._frequency_to_weight(vocab_freq) * self.vocab_freq_weight

            # General disambiguation: when time context is present and candidates mix
            # temporal-vs-opinion senses (e.g., 以降 vs 意見), prefer temporal terms.
            if temporal_disambiguation:
                if candidate in self.temporal_candidates:
                    score += self.temporal_context_bonus
                if candidate in self.opinion_candidates:
                    score -= self.temporal_context_penalty

            score += self._world_logic_adjustment(sentence, candidate)

            scores[candidate] = score

        if not scores:
            return None, 0

        best = max(scores.items(), key=lambda x: x[1])
        if best[1] > 0:
            return best[0], int(round(best[1]))

        return candidates[0] if candidates else None, 0

    def _resolve_learning_reading(self, wrong_word, correct_word, context_hint=""):
        """Resolve stable reading key for learned pairs from dictionary/context/MeCab fallbacks."""
        wrong = (wrong_word or "").strip()
        correct = (correct_word or "").strip()
        if not wrong or not correct:
            return ""

        # 1) Prefer readings already known by homophone map.
        reading_votes = []
        for token in (wrong, correct):
            info = self.homophones.get(token, {}) if token else {}
            for reading in info.get("readings", []) or []:
                rr = self._normalize_reading_key(reading)
                if rr:
                    reading_votes.append(rr)
        if reading_votes:
            return max(Counter(reading_votes).items(), key=lambda x: x[1])[0]

        # 2) Reuse existing reading groups containing either token.
        for reading, group in (self._reading_groups or {}).items():
            words = set(group.get("words", []) or [])
            if wrong in words or correct in words:
                return reading

        # 3) If context sentence is available, use token reading from sentence analysis.
        if context_hint:
            for token in self._analyze_morphemes(context_hint):
                surface = (token.get("surface") or "").strip()
                if surface not in {wrong, correct}:
                    continue
                rr = self._normalize_reading_key(token.get("reading", ""))
                if rr:
                    return rr

        # 4) Final fallback: infer from each token directly.
        for token in (correct, wrong):
            rr = self._normalize_reading_key(self._reading_for_text(token))
            if rr:
                return rr

        return ""

    def _add_learned_homophone_entry(self, wrong_word, correct_word, context_hint="", source="auto_learned", freq=1):
        """Append a learned homophone mapping to the main dictionary file."""
        wrong_word = (wrong_word or "").strip()
        correct_word = (correct_word or "").strip()
        if not wrong_word or not correct_word or wrong_word == correct_word:
            return False

        if len(wrong_word) < 2 or len(correct_word) < 2:
            return False
        if len(wrong_word) > 8 or len(correct_word) > 8:
            return False
        if re.search(r"[\s、。,.!?]", wrong_word) or re.search(r"[\s、。,.!?]", correct_word):
            return False
        if "|" in wrong_word or "|" in correct_word or "," in wrong_word or "," in correct_word:
            return False

        if not str(self.homophones_file).lower().endswith(".json"):
            return False

        try:
            payload = {}
            if os.path.exists(self.homophones_file):
                with open(self.homophones_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    payload = loaded

            reading = self._resolve_learning_reading(wrong_word, correct_word, context_hint=context_hint)
            if not reading:
                return False

            group = payload.get(reading)
            if not isinstance(group, dict):
                group = {
                    "default_candidate": correct_word,
                    "candidates": [],
                }
                payload[reading] = group

            candidates = group.get("candidates", [])
            if not isinstance(candidates, list):
                candidates = []
                group["candidates"] = candidates

            existing_words = set()
            existing_items = {}
            for item in candidates:
                if isinstance(item, dict):
                    w = str(item.get("word", "") or "").strip()
                    if w:
                        existing_words.add(w)
                        existing_items[w] = item

            # If this pair is already representable inside the same reading group,
            # persist verification signal by boosting confidence of correct word.
            if wrong_word in existing_words and correct_word in existing_words:
                updated_existing = False
                if source in {"llm_verified", "human_verified", "auto_runtime"}:
                    correct_item = existing_items.get(correct_word)
                    if isinstance(correct_item, dict):
                        old_weight = float(correct_item.get("weight", 1.0) or 1.0)
                        new_weight = max(old_weight, 1.5)
                        if new_weight > old_weight:
                            correct_item["weight"] = new_weight
                            updated_existing = True
                if not updated_existing:
                    return False

                os.makedirs(os.path.dirname(self.homophones_file), exist_ok=True)
                with open(self.homophones_file, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)

                self.homophones = self._load_homophone_dictionary(self.homophones_file)
                self._build_reading_index()
                return True

            updated = False

            if correct_word not in existing_words:
                candidates.append(
                    {
                        "word": correct_word,
                        "pos_tags": ["名詞"],
                        "context_rules": {
                            "require_any": [],
                            "require_collocation": [],
                            "exclude": [],
                        },
                        "weight": max(1.0, float(freq or 1) / 3.0),
                    }
                )
                existing_words.add(correct_word)
                updated = True

            if wrong_word not in existing_words:
                candidates.append(
                    {
                        "word": wrong_word,
                        "pos_tags": ["名詞"],
                        "context_rules": {
                            "require_any": [],
                            "require_collocation": [],
                            "exclude": [],
                        },
                        "weight": 1.0,
                    }
                )
                updated = True

            default_candidate = str(group.get("default_candidate", "") or "").strip()
            if not default_candidate:
                group["default_candidate"] = correct_word

            if not updated:
                return False

            os.makedirs(os.path.dirname(self.homophones_file), exist_ok=True)
            with open(self.homophones_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            self.homophones = self._load_homophone_dictionary(self.homophones_file)
            self._build_reading_index()
            return True
        except Exception:
            return False

    def _mark_llm_unavailable_if_rate_limited(self, error):
        """Disable future LLM calls in this run when rate-limit is hit."""
        if not self.disable_llm_on_rate_limit:
            return
        msg = str(error).lower()
        if "rate_limit" in msg or "rate limit" in msg:
            if not self._llm_unavailable:
                print("  [LLM] rate-limited, disable further LLM calls for this run")
            self._llm_unavailable = True

    def _reading_similarity(self, old_reading, new_reading):
        """Return reading similarity ratio in [0, 1]."""
        if not old_reading or not new_reading:
            return 0.0
        return SequenceMatcher(None, old_reading, new_reading).ratio()

    def _reading_similarity_robust(self, old_reading, new_reading, old_surface="", new_surface=""):
        """Compute robust reading similarity using raw/normalized/surface forms."""
        sims = []
        if old_reading and new_reading:
            sims.append(self._reading_similarity(old_reading, new_reading))
            sims.append(
                self._reading_similarity(
                    self._normalize_phonetic_text(old_reading),
                    self._normalize_phonetic_text(new_reading),
                )
            )
        if old_surface and new_surface:
            sims.append(
                self._reading_similarity(
                    self._normalize_phonetic_text(old_surface),
                    self._normalize_phonetic_text(new_surface),
                )
            )
        return max(sims) if sims else 0.0

    def _katakana_to_hiragana(self, text):
        """Convert katakana to hiragana for phonetic normalization."""
        result = []
        for ch in text or "":
            code = ord(ch)
            if 0x30A1 <= code <= 0x30F6:
                result.append(chr(code - 0x60))
            else:
                result.append(ch)
        return "".join(result)

    def _long_vowel_for_hiragana(self, ch):
        """Map hiragana char to its long-vowel base for handling ー."""
        if ch in "あかがさざただなはばぱまやらわぁゃ":
            return "あ"
        if ch in "いきぎしじちぢにひびぴみりゐぃ":
            return "い"
        if ch in "うくぐすずつづぬふぶぷむゆるぅゅ":
            return "う"
        if ch in "えけげせぜてでねへべぺめれゑぇ":
            return "え"
        if ch in "おこごそぞとのほぼぽもよろをぉょ":
            return "お"
        return ""

    def _normalize_phonetic_text(self, text):
        """Normalize script/long-vowel variation for robust homophone matching."""
        hira = self._katakana_to_hiragana((text or "").strip())
        out = []
        for ch in hira:
            if ch == "ー":
                if out:
                    vowel = self._long_vowel_for_hiragana(out[-1])
                    if vowel:
                        out.append(vowel)
                continue
            out.append(ch)
        normalized = "".join(out)
        # Common orthographic long-vowel variants.
        normalized = normalized.replace("おう", "おお")
        normalized = normalized.replace("えい", "ええ")
        normalized = re.sub(r"([こごそぞとのほぼぽもよろをお])う", r"\1お", normalized)
        normalized = re.sub(r"([けげせぜてでねへべぺめれえ])い", r"\1え", normalized)
        return normalized

    def _clean_reading_text(self, text):
        """Keep only Japanese phonetic chars from analyzer reading strings."""
        return re.sub(r"[^\u3040-\u309f\u30a0-\u30ffー]", "", (text or "").strip())

    def _is_short_kana_fragment(self, text):
        """Detect very short kana-only fragments that are often grammar particles."""
        value = (text or "").strip()
        if not value:
            return False
        if len(value) > 2:
            return False
        return bool(re.fullmatch(r"[\u3040-\u309f\u30a0-\u30ffー]+", value))

    def postprocess_spoken_style(self, sentence):
        """Normalize over-kanjified spoken forms to kana-first style for final output."""
        text = sentence or ""
        if not self.enable_post_normalization or not text:
            return text, []

        updated = text
        applied = []
        for src, dst in self.post_normalization_pairs:
            if src in updated:
                updated = updated.replace(src, dst)
                applied.append(f"{src}→{dst}")

        if updated != text:
            self.post_normalized_sentences += 1
        return updated, applied

    def _pos_tag_variants(self, tags):
        """Expand POS tags with coarse prefixes for robust compatibility checks."""
        variants = set()
        for raw in tags or []:
            tag = (raw or "").strip()
            if not tag:
                continue
            variants.add(tag)
            if "-" in tag:
                variants.add(tag.split("-", 1)[0])
        return variants

    def _is_pos_rewrite_allowed(self, source_pos, target_pos):
        """Enforce POS allow/deny policy from config."""
        source_variants = self._pos_tag_variants(source_pos)
        target_variants = self._pos_tag_variants(target_pos)

        if self.pos_deny:
            if source_variants.intersection(self.pos_deny) or target_variants.intersection(self.pos_deny):
                return False
        if self.pos_allow:
            source_ok = (not source_variants) or bool(source_variants.intersection(self.pos_allow))
            target_ok = (not target_variants) or bool(target_variants.intersection(self.pos_allow))
            return source_ok and target_ok
        return True

    def _init_morph_analyzer(self):
        """Initialize MeCab for POS-based filtering if available."""
        if not self.enable_pos_filter:
            return

        # Prefer helper wrapper for consistent token fields across dictionaries.
        try:
            self._morph_analyzer = JapaneseMorphAnalyzer(dicdir=self.mecab_dicdir)
            if self._morph_analyzer.available:
                self._mecab_tagger = self._morph_analyzer.tagger
                if MeCab is not None:
                    try:
                        if self.mecab_dicdir:
                            self._mecab_yomi_tagger = MeCab.Tagger(f"-d {self.mecab_dicdir} -Oyomi")
                        else:
                            self._mecab_yomi_tagger = MeCab.Tagger("-Oyomi")
                        self._mecab_yomi_tagger.parse("")
                    except Exception:
                        self._mecab_yomi_tagger = None
                print("  [POS] MeCab helper enabled")
                return
        except Exception as e:
            self._morph_analyzer = None
            self._mecab_yomi_tagger = None
            print(f"  [POS] MeCab helper unavailable, try direct tagger: {e}")

        if MeCab is None:
            return
        try:
            self._mecab_tagger = MeCab.Tagger()
            # Warm up parser to avoid occasional first-call issues.
            self._mecab_tagger.parse("")
            try:
                if self.mecab_dicdir:
                    self._mecab_yomi_tagger = MeCab.Tagger(f"-d {self.mecab_dicdir} -Oyomi")
                else:
                    self._mecab_yomi_tagger = MeCab.Tagger("-Oyomi")
                self._mecab_yomi_tagger.parse("")
            except Exception:
                self._mecab_yomi_tagger = None
            print("  [POS] MeCab direct tagger enabled")
        except Exception as e:
            self._mecab_tagger = None
            self._mecab_yomi_tagger = None
            print(f"  [POS] MeCab unavailable, fallback to current logic: {e}")

    def _analyze_morphemes(self, text):
        """Return morphemes with character spans and coarse POS."""
        if not text or not self._mecab_tagger:
            return []
        if text in self._morpheme_cache:
            return self._morpheme_cache[text]

        if self._morph_analyzer and self._morph_analyzer.available:
            nodes = []
            cursor = 0
            for token in self._morph_analyzer.tokenize(text):
                surface = (token.get("surface") or "").strip()
                if not surface:
                    continue
                start = text.find(surface, cursor)
                if start < 0:
                    start = cursor
                end = start + len(surface)
                pos = (token.get("pos") or "").strip()
                pos_parts = [pos]
                for detail_key in ("pos_detail1", "pos_detail2", "pos_detail3"):
                    detail = (token.get(detail_key) or "").strip()
                    if detail and detail != "*":
                        pos_parts.append(detail)
                pos_full = "-".join([p for p in pos_parts if p])
                nodes.append(
                    {
                        "surface": surface,
                        "start": start,
                        "end": end,
                        "pos": pos,
                        "pos_full": pos_full,
                        "reading": (token.get("reading") or token.get("pronunciation") or "").strip(),
                    }
                )
                cursor = end
            self._morpheme_cache[text] = nodes
            return nodes

        nodes = []
        cursor = 0
        node = self._mecab_tagger.parseToNode(text)
        while node:
            surface = node.surface or ""
            if surface:
                start = text.find(surface, cursor)
                if start < 0:
                    start = cursor
                end = start + len(surface)
                features = (node.feature or "").split(",")
                pos = features[0].strip() if features else ""
                pos_parts = [pos]
                for idx in (1, 2, 3):
                    if len(features) > idx:
                        detail = (features[idx] or "").strip()
                        if detail and detail != "*":
                            pos_parts.append(detail)
                pos_full = "-".join([p for p in pos_parts if p])
                reading = ""
                if len(features) > 7:
                    reading = (features[7] or "").strip()
                nodes.append(
                    {
                        "surface": surface,
                        "start": start,
                        "end": end,
                        "pos": pos,
                        "pos_full": pos_full,
                        "reading": reading,
                    }
                )
                cursor = end
            node = node.next
        self._morpheme_cache[text] = nodes
        return nodes

    def _reading_for_text(self, text):
        """Get concatenated MeCab reading for a short text, cached."""
        key = text or ""
        if key in self._reading_cache:
            return self._reading_cache[key]
        if not key or not self._mecab_tagger:
            self._reading_cache[key] = ""
            return ""

        readings = []
        for token in self._analyze_morphemes(key):
            reading = self._clean_reading_text(token.get("reading", ""))
            if reading and reading != "*":
                readings.append(reading)
            else:
                surface = self._clean_reading_text(token.get("surface", ""))
                readings.append(surface or token.get("surface", ""))
        value = "".join(readings)
        cleaned = self._clean_reading_text(value)

        # Some dictionaries return non-phonetic or partial readings for rare tokens.
        # Use -Oyomi as a fallback for kanji-heavy fragments.
        has_kanji = bool(re.search(r"[\u4e00-\u9fff々ヶ]", key))
        looks_partial = has_kanji and len(cleaned) < max(2, int(len(key) * 0.6))
        if (not cleaned or looks_partial) and self._mecab_yomi_tagger:
            try:
                yomi_raw = (self._mecab_yomi_tagger.parse(key) or "").strip()
                if yomi_raw:
                    yomi_line = yomi_raw.splitlines()[0].strip()
                    yomi_clean = self._clean_reading_text(yomi_line)
                    if yomi_clean:
                        cleaned = yomi_clean
            except Exception:
                pass

        value = cleaned or value
        self._reading_cache[key] = value
        return value

    def _is_reading_compatible(self, old_fragment, new_fragment):
        """Check homophone compatibility using MeCab readings when available."""
        if not self.no_dict_require_reading_match or not self.check_reading or not self._mecab_tagger:
            return True
        # If dictionary explicitly links the pair, trust that signal.
        if self._has_explicit_homophone_signal(old_fragment, new_fragment):
            return True
        old_reading = self._reading_for_text(old_fragment)
        new_reading = self._reading_for_text(new_fragment)
        if not old_reading or not new_reading:
            return True
        similarity = self._reading_similarity_robust(
            old_reading,
            new_reading,
            old_surface=old_fragment,
            new_surface=new_fragment,
        )
        return similarity >= self.strict_reading_ratio

    def _pos_for_span(self, text, start, length):
        """Collect POS tags overlapping the span in a sentence."""
        end = start + max(0, length)
        if end <= start:
            return set()
        pos_tags = set()
        for token in self._analyze_morphemes(text):
            if token["end"] <= start or token["start"] >= end:
                continue
            if token["pos"]:
                pos_tags.add(token["pos"])
            if token.get("pos_full"):
                pos_tags.add(token.get("pos_full"))
        return pos_tags

    def _candidate_pos(self, text):
        """Get POS tags for a candidate term (cached)."""
        key = text or ""
        if key in self._candidate_pos_cache:
            return self._candidate_pos_cache[key]
        pos_tags = set()
        for token in self._analyze_morphemes(key):
            if token["pos"]:
                pos_tags.add(token["pos"])
            if token.get("pos_full"):
                pos_tags.add(token.get("pos_full"))
        self._candidate_pos_cache[key] = pos_tags
        return pos_tags

    def _filter_candidates_by_pos(self, sentence, word, start, candidates, candidate_pos_hints=None):
        """Filter dictionary candidates by POS compatibility with source span."""
        if not self._mecab_tagger:
            return candidates
        source_pos = self._pos_for_span(sentence, start, len(word))
        if not source_pos:
            return candidates

        filtered = []
        removed = 0
        candidate_pos_hints = candidate_pos_hints or {}
        for candidate in candidates:
            hint_tags = candidate_pos_hints.get(candidate, [])
            if isinstance(hint_tags, str):
                hint_tags = [hint_tags]
            candidate_pos = set([str(t).strip() for t in (hint_tags or []) if str(t).strip()])
            if not candidate_pos:
                candidate_pos = self._candidate_pos(candidate)
            if not self._is_pos_rewrite_allowed(source_pos, candidate_pos):
                removed += 1
                self._log_pos_prune(sentence, word, candidate, "pos_allow_deny", source_pos, candidate_pos)
                continue

            source_variants = self._pos_tag_variants(source_pos)
            candidate_variants = self._pos_tag_variants(candidate_pos)
            if not candidate_variants or source_variants.intersection(candidate_variants):
                filtered.append(candidate)
            else:
                removed += 1
                self._log_pos_prune(sentence, word, candidate, "pos_mismatch", source_pos, candidate_pos)

        if filtered:
            self.pos_filtered_candidates += removed
            return filtered
        return candidates

    def _passes_no_dict_pos_guard(self, sentence, old_fragment, new_fragment):
        """Reject free-form replacements when POS clearly mismatches."""
        if not self.no_dict_enable_pos_guard:
            return True
        if not self._mecab_tagger:
            return True
        start = sentence.find(old_fragment)
        if start < 0:
            return True

        tokens = self._analyze_morphemes(sentence)
        if self.no_dict_token_boundary_check and tokens:
            span_end = start + len(old_fragment)
            has_start_boundary = any(t["start"] == start for t in tokens)
            has_end_boundary = any(t["end"] == span_end for t in tokens)
            if not (has_start_boundary and has_end_boundary):
                self._log_pos_prune(sentence, old_fragment, new_fragment, "token_boundary_guard")
                return False

        source_pos = self._pos_for_span(sentence, start, len(old_fragment))
        target_pos = self._candidate_pos(new_fragment)
        if not self._is_pos_rewrite_allowed(source_pos, target_pos):
            self._log_pos_prune(sentence, old_fragment, new_fragment, "pos_allow_deny_guard", source_pos, target_pos)
            return False
        if source_pos and target_pos and not source_pos.intersection(target_pos):
            self._log_pos_prune(sentence, old_fragment, new_fragment, "pos_mismatch_guard", source_pos, target_pos)
            return False
        if not self._is_reading_compatible(old_fragment, new_fragment):
            self._log_pos_prune(sentence, old_fragment, new_fragment, "reading_mismatch_guard", source_pos, target_pos)
            return False
        return True

    def _passes_dict_rewrite_guard(self, sentence, old_fragment, new_fragment):
        """Guard dictionary/candidate replacement without strict no-dict span constraints."""
        if not self.enable_pos_filter:
            return True
        if not self._mecab_tagger:
            return True

        start = sentence.find(old_fragment)
        if start < 0:
            return True

        source_pos = self._pos_for_span(sentence, start, len(old_fragment))
        target_pos = self._candidate_pos(new_fragment)
        explicit_homophone = self._has_explicit_homophone_signal(old_fragment, new_fragment)
        reading_ok = self._is_reading_compatible(old_fragment, new_fragment)

        if not self._is_pos_rewrite_allowed(source_pos, target_pos):
            if explicit_homophone and reading_ok:
                return True
            self._log_pos_prune(sentence, old_fragment, new_fragment, "dict_pos_allow_deny_guard", source_pos, target_pos)
            return False
        if source_pos and target_pos and not source_pos.intersection(target_pos):
            if explicit_homophone and reading_ok:
                return True
            self._log_pos_prune(sentence, old_fragment, new_fragment, "dict_pos_mismatch_guard", source_pos, target_pos)
            return False

        # Domain plausibility guard: block rewrites that make active-domain semantics worse.
        old_logic = self._world_logic_adjustment(sentence, old_fragment)
        new_logic = self._world_logic_adjustment(sentence, new_fragment)
        if new_logic < 0 and new_logic < old_logic:
            self._log_pos_prune(sentence, old_fragment, new_fragment, "dict_world_logic_guard", source_pos, target_pos)
            return False

        # Generic frequency sanity guard: avoid replacing with an extremely rarer token
        # unless there is no reliable frequency evidence.
        old_freq = self._frequency_to_int(self._get_vocab_frequency(old_fragment))
        new_freq = self._frequency_to_int(self._get_vocab_frequency(new_fragment))
        if old_freq > 0 and new_freq > 0:
            if new_freq < max(1, int(old_freq * 0.1)):
                self._log_pos_prune(
                    sentence,
                    old_fragment,
                    new_fragment,
                    "dict_frequency_guard",
                    source_pos,
                    target_pos,
                )
                return False
        return True

    def _remember_verified_mapping(self, wrong_word, correct_word):
        """Remember LLM-verified mapping for immediate reuse in current run."""
        if not self.use_verified_runtime_map:
            return
        if self._is_blacklisted_mapping(wrong_word, correct_word):
            return
        key = (wrong_word, correct_word)
        hits = self._verified_mapping_hits.get(key, 0) + 1
        self._verified_mapping_hits[key] = hits
        if hits >= self.verified_runtime_min_hits:
            prev = self._verified_runtime_map.get(wrong_word)
            if prev != correct_word:
                self._verified_runtime_map[wrong_word] = correct_word
                self.runtime_map_learned += 1
        if hits >= self.verified_runtime_persist_min_hits:
            self._persist_verified_runtime_state()
            if self.auto_learn_apply_on_verified_hits:
                self._add_learned_homophone_entry(
                    wrong_word,
                    correct_word,
                    context_hint="runtime_verified",
                    source="llm_verified",
                    freq=hits,
                )

    def _apply_verified_runtime_map(self, sentence):
        """Apply verified mappings before dictionary/LLM to reduce repeated mistakes."""
        if not self.use_verified_runtime_map or not self._verified_runtime_map:
            return sentence, []

        updated = sentence
        changes = []
        for wrong_word in sorted(self._verified_runtime_map.keys(), key=lambda x: len(x), reverse=True):
            correct_word = self._verified_runtime_map[wrong_word]
            if not wrong_word or not correct_word or wrong_word == correct_word:
                continue
            start = updated.find(wrong_word)
            if start < 0:
                continue
            if self.no_dict_token_boundary_check and not self._is_token_boundary_match(updated, start, len(wrong_word)):
                continue
            if not self._passes_no_dict_pos_guard(updated, wrong_word, correct_word):
                continue
            if not self._passes_dict_rewrite_guard(updated, wrong_word, correct_word):
                continue
            updated = updated.replace(wrong_word, correct_word, 1)
            changes.append(f"{wrong_word}→{correct_word}")

        if changes:
            self.runtime_map_applied += len(changes)
        return updated, changes

    def _strip_reasoning_blocks(self, text):
        """Remove reasoning tags often emitted by some models before strict parsing."""
        if not text:
            return ""
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        # Handle truncated responses where <think> is opened but never closed.
        cleaned = re.sub(r"<think>[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _normalize_pre_no_dict_sentence(self, sentence):
        """Apply generic orthographic normalization before no-dict detection/proposal."""
        text = sentence or ""
        if not self.no_dict_enable_pre_normalization or not text:
            return text, []

        updated = text
        applied = []
        for pattern, repl in self.no_dict_pre_normalization_rules:
            try:
                updated, count = re.subn(pattern, repl, updated)
            except re.error:
                continue
            if count > 0:
                applied.append(f"{pattern}=>{repl} x{count}")

        if updated != text:
            self.no_dict_pre_normalized_sentences += 1
        return updated, applied

    def _detect_no_dict_error(self, sentence):
        """Step-1 gate: detect whether sentence likely contains a clear homophone ASR error."""
        prompt = f"""【日本語ASR同音誤り判定】

入力文: {sentence}

【判定ルール】
    1. 同音・近音の語彙誤りが明確な場合のみ ERROR
    2. 前後3〜5語の連語（collocation）を見て、文脈上の不自然さがあるときだけ ERROR
    3. 意味の言い換え・敬語変換・文体変換・句読点補正は対象外（必ず OK）
    4. 不確実なら必ず OK
    5. 説明は禁止

【出力形式】
- DECISION: ERROR | TARGET: 誤り語
- DECISION: OK
"""
        try:
            response = self._call_llm_with_retry(
                prompt,
                max_tokens=40,
                client=self.no_dict_client,
                model=self.no_dict_model,
            )
            raw = self._strip_reasoning_blocks(response.choices[0].message.content)
            upper = raw.upper()
            if "DECISION:" not in upper or "ERROR" not in upper:
                return False, ""

            target_match = re.search(r"TARGET\s*:\s*(.+)", raw, flags=re.IGNORECASE)
            target = target_match.group(1).strip() if target_match else ""
            if target:
                target = target.replace("<", "").replace(">", "").strip()
            return True, target
        except Exception:
            return False, ""

    def _detect_pos_anomaly_target(self, sentence):
        """Find a suspicious token by simple POS-pattern anomalies for no-dict prioritization."""
        if not self.prioritize_pos_anomaly:
            return ""
        tokens = self._analyze_morphemes(sentence)
        if not tokens:
            return ""

        for i, token in enumerate(tokens):
            surface = (token.get("surface") or "").strip()
            pos = token.get("pos", "")
            if not surface:
                continue

            # Kanji token tagged as function word is often an ASR confusion signal.
            if re.search(r"[\u4e00-\u9fff々ヶ]", surface) and pos in {"助詞", "助動詞", "記号"}:
                return surface

            # Isolated single-kanji noun between function words is frequently suspicious.
            if len(surface) == 1 and re.search(r"[\u4e00-\u9fff々ヶ]", surface) and pos == "名詞":
                prev_pos = tokens[i - 1].get("pos", "") if i > 0 else ""
                next_pos = tokens[i + 1].get("pos", "") if i + 1 < len(tokens) else ""
                if prev_pos in {"助詞", "助動詞"} and next_pos in {"助詞", "助動詞"}:
                    return surface

        return ""

    def _extract_single_span_change(self, original_sentence, corrected_sentence):
        """Extract one contiguous replacement span between original and corrected strings."""
        if original_sentence == corrected_sentence:
            return None, None

        prefix = 0
        max_prefix = min(len(original_sentence), len(corrected_sentence))
        while prefix < max_prefix and original_sentence[prefix] == corrected_sentence[prefix]:
            prefix += 1

        suffix = 0
        orig_remain = len(original_sentence) - prefix
        corr_remain = len(corrected_sentence) - prefix
        max_suffix = min(orig_remain, corr_remain)
        while suffix < max_suffix and original_sentence[len(original_sentence) - 1 - suffix] == corrected_sentence[len(corrected_sentence) - 1 - suffix]:
            suffix += 1

        old_fragment = original_sentence[prefix:len(original_sentence) - suffix if suffix > 0 else len(original_sentence)]
        new_fragment = corrected_sentence[prefix:len(corrected_sentence) - suffix if suffix > 0 else len(corrected_sentence)]
        return old_fragment, new_fragment

    def _passes_no_dict_span_guard(self, old_fragment, new_fragment):
        """Hard constraints for no-dict acceptance to avoid sentence-level rewrites."""
        if not self.no_dict_enable_span_guard:
            return True
        if not old_fragment or not new_fragment:
            return False
        if old_fragment == new_fragment:
            return False
        if len(old_fragment) > self.no_dict_max_span_len or len(new_fragment) > self.no_dict_max_span_len:
            return False
        if abs(len(old_fragment) - len(new_fragment)) > self.no_dict_max_len_delta:
            return False
        if re.search(r"[\s、。,.!?]", old_fragment) or re.search(r"[\s、。,.!?]", new_fragment):
            return False
        if self.no_dict_require_kanji:
            if not (re.search(r"[\u4e00-\u9fff]", old_fragment) or re.search(r"[\u4e00-\u9fff]", new_fragment)):
                return False
        return True

    def _has_explicit_homophone_signal(self, old_fragment, new_fragment):
        """Check whether old->new has explicit support from dictionary or reading index."""
        old = (old_fragment or "").strip()
        new = (new_fragment or "").strip()
        if not old or not new or old == new:
            return False

        old_info = self.homophones.get(old, {}) or {}
        old_candidates = set(old_info.get("candidates", []) or [])
        if new in old_candidates:
            return True

        old_reading = self._reading_for_text(old)
        if old_reading:
            indexed = self._reading_candidate_map.get(old_reading, set())
            if new in indexed:
                return True

        new_info = self.homophones.get(new, {}) or {}
        new_candidates = set(new_info.get("candidates", []) or [])
        if old in new_candidates:
            return True

        new_reading = self._reading_for_text(new)
        if new_reading:
            indexed = self._reading_candidate_map.get(new_reading, set())
            if old in indexed:
                return True

        return False

    def _passes_no_dict_homophone_gate(self, sentence, old_fragment, new_fragment):
        """Strict no-dict gate: accept only clear homophone-like replacements."""
        if not old_fragment or not new_fragment or old_fragment == new_fragment:
            return False
        # Always enforce compact lexical spans in no-dict mode to avoid phrase-level drift.
        if len(old_fragment) > self.no_dict_max_span_len or len(new_fragment) > self.no_dict_max_span_len:
            return False
        if abs(len(old_fragment) - len(new_fragment)) > self.no_dict_max_len_delta:
            return False
        if re.search(r"[\s、。,.!?]", old_fragment) or re.search(r"[\s、。,.!?]", new_fragment):
            return False
        if not self._passes_no_dict_span_guard(old_fragment, new_fragment):
            return False

        start = sentence.find(old_fragment)
        if self.no_dict_token_boundary_check and start >= 0:
            if not self._is_token_boundary_match(sentence, start, len(old_fragment)):
                return False

        old_reading = self._reading_for_text(old_fragment)
        new_reading = self._reading_for_text(new_fragment)
        if old_reading and new_reading:
            similarity = self._reading_similarity_robust(
                old_reading,
                new_reading,
                old_surface=old_fragment,
                new_surface=new_fragment,
            )
            if similarity < self.no_dict_min_reading_similarity:
                return False
        elif self.no_dict_require_reading_match:
            return False

        if not self._passes_no_dict_pos_guard(sentence, old_fragment, new_fragment):
            return False

        # Domain world-logic: block implausible content words in active contexts.
        if self._world_logic_adjustment(sentence, new_fragment) < 0:
            return False

        if self.no_dict_require_explicit_homophone and not self._has_explicit_homophone_signal(old_fragment, new_fragment):
            # Allow OOV fallback only for very likely homophone substitutions.
            if not self.no_dict_allow_oov_homophone:
                return False
            # OOV fallback should also work for kana/katakana homophones.
            if not (
                re.search(r"[\u3040-\u30ff\u4e00-\u9fff々ヶー]", old_fragment)
                and re.search(r"[\u3040-\u30ff\u4e00-\u9fff々ヶー]", new_fragment)
            ):
                return False
            if max(len(old_fragment), len(new_fragment)) > 4:
                return False
            if old_reading and new_reading:
                if self._reading_similarity_robust(
                    old_reading,
                    new_reading,
                    old_surface=old_fragment,
                    new_surface=new_fragment,
                ) < max(self.no_dict_min_reading_similarity, 0.97):
                    return False
            else:
                return False

        return True

    def _propose_no_dict_replacement(self, sentence, target_hint="", force=False):
        """Step-2 gate: propose a single homophone lexical replacement instead of rewriting whole sentence."""
        hint_line = f"候補誤り語: {target_hint}\n" if target_hint else ""
        candidate_line = ""
        if target_hint:
            hint_reading = self._reading_for_text(target_hint)
            if hint_reading:
                raw_candidates = sorted(self._reading_candidate_map.get(hint_reading, set()))
                filtered = [c for c in raw_candidates if c and c != target_hint][:8]
                if filtered:
                    candidate_line = "RIGHT候補(必ずこの中から1つ選ぶ): " + " / ".join(filtered) + "\n"
        base_prompt = self.no_dict_prompt_template or (
            "あなたはASR（音声認識）の同音異義語・近音語の誤り修正を行う専門AIです。\n"
            "出力は必ず <改>[...] の1行のみです。"
        )
        prompt = (
            f"{base_prompt}\n\n"
            f"【補助情報】\n"
            f"{hint_line}"
            f"{candidate_line}"
            f"入力: {sentence}\n"
            f"出力:"
        )
        try:
            response = self._call_llm_with_retry(
                prompt,
                max_tokens=400,
                client=self.no_dict_client,
                model=self.no_dict_model,
                system_message=(
                    "You are a Japanese ASR error correction assistant. "
                    "Output in one line only as <改>[corrected sentence]. "
                    "DO NOT use <think> tags, <reasoning>, or any internal monologue. "
                    "No explanations, no extra text."
                ),
            )
            raw = self._strip_reasoning_blocks(response.choices[0].message.content)
            match = re.search(r"<改>\[(.*?)\]", raw or "", flags=re.DOTALL)
            if not match:
                return None, None

            corrected_sentence = match.group(1).strip()
            if not corrected_sentence or corrected_sentence == sentence:
                return None, None

            wrong, right = self._extract_single_span_change(sentence, corrected_sentence)
            if not wrong or not right:
                return None, None
            print(f"  [PARSED] WRONG={wrong!r} RIGHT={right!r}")
            if not wrong or not right or wrong == right:
                print(f"  [REJECT] Empty or same pair")
                return None, None
            # Avoid overly aggressive single-character and phrase-level rewrites.
            if len(wrong) < 2 or len(right) < 2:
                # Allow strict one-kanji substitutions to recover short ASR confusions.
                if not (
                    len(wrong) == 1
                    and len(right) == 1
                    and re.search(r"[\u4e00-\u9fff]", wrong)
                    and re.search(r"[\u4e00-\u9fff]", right)
                ):
                    print(f"  [REJECT] Length too short: {len(wrong)},{len(right)}")
                    return None, None
            if len(wrong) > 8 or len(right) > 8:
                print(f"  [REJECT] Length too long: {len(wrong)},{len(right)}")
                return None, None
            if not re.fullmatch(r"[\u3040-\u30ff\u4e00-\u9fffーA-Za-z0-9々ヶ]+", wrong):
                print(f"  [REJECT] WRONG has invalid chars")
                return None, None
            if not re.fullmatch(r"[\u3040-\u30ff\u4e00-\u9fffーA-Za-z0-9々ヶ]+", right):
                print(f"  [REJECT] RIGHT has invalid chars")
                return None, None
            if wrong not in sentence:
                print(f"  [REJECT] WRONG not in sentence")
                return None, None
            start = sentence.find(wrong)
            if start < 0:
                print(f"  [REJECT] WRONG position not found")
                return None, None
            # Keep replacements aligned to token boundaries to avoid partial-word drift.
            if not self._is_token_boundary_match(sentence, start, len(wrong)):
                print(f"  [REJECT] Token boundary mismatch")
                return None, None
            if abs(len(wrong) - len(right)) > self.no_dict_max_len_delta:
                print(f"  [REJECT] Length delta too large: {abs(len(wrong) - len(right))} > {self.no_dict_max_len_delta}")
                return None, None
            old_reading = self._reading_for_text(wrong)
            new_reading = self._reading_for_text(right)
            print(f"  [READING] old={old_reading!r} new={new_reading!r}")
            sim = self._reading_similarity_robust(
                old_reading,
                new_reading,
                old_surface=wrong,
                new_surface=right,
            )
            print(f"  [SIM] {sim:.3f} vs {self.no_dict_adaptive_similarity_floor}")
            if sim < self.no_dict_adaptive_similarity_floor:
                print(f"  [REJECT] Reading similarity too low: {sim:.3f} < {self.no_dict_adaptive_similarity_floor}")
                return None, None
            if len(wrong) == 1 and len(right) == 1:
                if not (
                    re.search(r"[\u4e00-\u9fff]", wrong)
                    and re.search(r"[\u4e00-\u9fff]", right)
                ):
                    print(f"  [REJECT] Single kanji but one is not kanji")
                    return None, None
                if sim < 0.90:
                    print(f"  [REJECT] Single kanji but similarity too low: {sim:.3f} < 0.90")
                    return None, None
            print(f"  [ACCEPT] {wrong} -> {right}")
            return wrong, right
        except Exception as e:
            if "rate" in str(e).lower():
                pass  # Silently handle rate limits (logged elsewhere)
            else:
                print(f"[PROPOSER-ERROR] {type(e).__name__}: {e}")
            return None, None









    def _maybe_auto_learn_from_no_dict(self, old_fragment, new_fragment):
        """Queue no-dict corrections for later label-based verification."""
        if not self.no_dict_auto_learn_from_llm:
            return
        if not old_fragment or not new_fragment or old_fragment == new_fragment:
            return
        if self._is_blacklisted_mapping(old_fragment, new_fragment):
            return

        old_reading = self._reading_for_text(old_fragment)
        new_reading = self._reading_for_text(new_fragment)
        similarity = self._reading_similarity_robust(
            old_reading,
            new_reading,
            old_surface=old_fragment,
            new_surface=new_fragment,
        )
        if similarity < self.no_dict_auto_learn_min_similarity:
            return

        key = (old_fragment, new_fragment)
        hits = int(self._pending_no_dict_pairs.get(key, 0) or 0) + 1
        self._pending_no_dict_pairs[key] = hits
        print(f"  [NO-DICT-PENDING] {old_fragment} -> {new_fragment} (hits={hits})")

    def _pick_aggressive_no_dict_candidate(self, sentence, target_hint):
        """Pick a fallback replacement from reading-group candidates when KEEP is disabled."""
        if not sentence or not target_hint:
            return ""
        if target_hint not in sentence:
            return ""

        reading = self._reading_for_text(target_hint)
        if not reading:
            return ""

        raw_candidates = sorted(self._reading_candidate_map.get(reading, set()))
        candidates = [c for c in raw_candidates if c and c != target_hint]
        if not candidates:
            return ""

        left_tokens, right_tokens, _, _ = self._extract_scoring_context_window(sentence, target_hint, window_size=3)
        embedding_scores = self._embedding_delta_scores(sentence, target_hint, candidates, client=self.no_dict_client)
        old_reading = self._reading_for_text(target_hint)

        ranked = []
        ranked_loose = []
        for candidate in candidates:
            score = self._collocation_score(target_hint, candidate, left_tokens, right_tokens) * self.collocation_weight
            if embedding_scores:
                score += float(embedding_scores.get(candidate, 0.0) or 0.0) * self.embedding_score_weight
            new_reading = self._reading_for_text(candidate)
            sim_bonus = 0.0
            if old_reading and new_reading:
                sim_bonus = self._reading_similarity_robust(
                    old_reading,
                    new_reading,
                    old_surface=target_hint,
                    new_surface=candidate,
                )
                score += sim_bonus * 2.0
            ranked_loose.append((score, sim_bonus, candidate))
            if self._passes_no_dict_homophone_gate(sentence, target_hint, candidate):
                ranked.append((score, candidate))

        if ranked:
            ranked.sort(key=lambda x: x[0], reverse=True)
            return ranked[0][1]

        if self.force_change_on_llm_uncertain and ranked_loose:
            ranked_loose.sort(key=lambda x: x[0], reverse=True)
            best_score, best_sim, best_candidate = ranked_loose[0]
            if best_sim >= 0.45:
                return best_candidate
        return ""

    def llm_correct_without_dictionary(self, sentence, target_hint=""):
        """No-dict stage: ask LLM for one homophone replacement and return sentence-level result."""
        if self._llm_unavailable:
            return sentence

        # Step 1: Detect if sentence has an error (optional, can be expensive)
        should_fix = bool(target_hint)
        
        if self.enable_no_dict_detector:
            should_fix, target_hint = self._detect_no_dict_error(sentence)
        
        # Step 2: Try POS anomaly detection as fallback/alternative
        if not should_fix and self.prioritize_pos_anomaly:
            anomaly_target = self._detect_pos_anomaly_target(sentence)
            if anomaly_target:
                should_fix = True
                target_hint = anomaly_target

        # Skip no-dict LLM if detector does not find a concrete signal.
        if not should_fix:
            return sentence

        self.no_dict_llm_calls += 1

        try:
            wrong, right = self._propose_no_dict_replacement(sentence, target_hint=target_hint, force=True)
        except Exception as e:
            self._mark_llm_unavailable_if_rate_limited(e)
            print(f"LLM no-dict error: {e}")
            return sentence

        if not wrong or not right:
            if self.disable_keep_choice:
                forced = self._pick_aggressive_no_dict_candidate(sentence, target_hint)
                if forced:
                    wrong, right = target_hint, forced
                else:
                    return sentence
            else:
                return sentence

        source = wrong
        if target_hint and target_hint in sentence:
            source = target_hint
        elif source not in sentence and target_hint and target_hint in sentence:
            source = target_hint
        if source not in sentence:
            return sentence

        if self.no_dict_avoid_short_words and (
            self._is_short_kana_fragment(source) or self._is_short_kana_fragment(right)
        ):
            self.no_dict_short_rejects += 1
            self.no_dict_llm_rejected += 1
            return sentence

        if not self._passes_no_dict_pos_guard(sentence, source, right):
            if not (self.disable_keep_choice and self.force_change_on_llm_uncertain):
                self.no_dict_llm_rejected += 1
                return sentence

        if not self._passes_no_dict_homophone_gate(sentence, source, right):
            if not (self.disable_keep_choice and self.force_change_on_llm_uncertain):
                self.no_dict_llm_rejected += 1
                return sentence
            old_reading = self._reading_for_text(source)
            new_reading = self._reading_for_text(right)
            if not old_reading or not new_reading:
                self.no_dict_llm_rejected += 1
                return sentence
            relaxed_sim = self._reading_similarity_robust(
                old_reading,
                new_reading,
                old_surface=source,
                new_surface=right,
            )
            if relaxed_sim < max(0.50, self.no_dict_min_reading_similarity - 0.10):
                self.no_dict_llm_rejected += 1
                return sentence
            if abs(len(source) - len(right)) > max(self.no_dict_max_len_delta, 4):
                self.no_dict_llm_rejected += 1
                return sentence

        if self._is_blacklisted_mapping(source, right):
            self.no_dict_llm_rejected += 1
            return sentence

        return sentence.replace(source, right, 1)

    def learn_from_feedback(self, original_sentence, corrected_sentence, expected_sentence):
        """Learn only from no-dict pairs verified by label feedback."""
        if not self.auto_learn_dictionary:
            return False
        if not expected_sentence:
            return False
        if not self._last_no_dict_pairs:
            return False

        pair_candidates = []
        for wrong_word, correct_word in self._last_no_dict_pairs or []:
            wrong = (wrong_word or "").strip()
            correct = (correct_word or "").strip()
            if wrong and correct and wrong != correct:
                pair_candidates.append((wrong, correct))

        if not pair_candidates:
            return False

        learned_any = False
        seen = set()
        for wrong_word, correct_word in pair_candidates:
            key = (wrong_word, correct_word)
            if key in seen:
                continue
            seen.add(key)
            hits = max(1, int(self._pending_no_dict_pairs.pop(key, 1) or 1))

            is_verified = (
                corrected_sentence == expected_sentence
                and correct_word in expected_sentence
                and wrong_word not in expected_sentence
            )

            if not is_verified:
                if self.blacklist_on_incorrect_feedback:
                    self._append_runtime_blacklist(
                        wrong_word,
                        correct_word,
                        reason="no_dict_feedback_incorrect",
                        context_original=original_sentence,
                        context_corrected=corrected_sentence,
                        context_expected=expected_sentence,
                    )
                    self._drop_runtime_mapping(wrong_word, correct_word)
                    print(f"  [NO-DICT-BLACKLIST] {wrong_word} -> {correct_word}")
                else:
                    print(f"  [NO-DICT-SKIP] {wrong_word} -> {correct_word} (incorrect feedback, not blacklisted)")
                continue

            if len(wrong_word) < self.auto_learn_min_len or len(correct_word) < self.auto_learn_min_len:
                continue
            if not self._is_reading_compatible(wrong_word, correct_word):
                continue
            if self._is_blacklisted_mapping(wrong_word, correct_word):
                continue

            self._remember_verified_mapping(wrong_word, correct_word)
            self.no_dict_pair_verified_accept += 1

            added = self._add_learned_homophone_entry(
                wrong_word,
                correct_word,
                context_hint=expected_sentence,
                source="no_dict_verified",
                freq=hits,
            )
            if added:
                learned_any = True
                self.no_dict_auto_learned += 1
                print(f"  [NO-DICT-LEARNED] {wrong_word} -> {correct_word} (hits={hits})")

        if learned_any:
            self._persist_verified_runtime_state()
        return learned_any

    def _call_llm_with_retry(self, prompt, max_tokens=20, client=None, model=None, system_message=None):
        """Call LLM with retry/backoff for transient API failures."""
        use_client = client or self.client
        use_model = model or self.model
        last_error = None
        for attempt in range(self.max_api_retries + 1):
            try:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})

                is_groq_client = Groq is not None and isinstance(use_client, Groq)
                base_kwargs = {
                    "model": use_model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                }
                if is_groq_client:
                    base_kwargs["max_completion_tokens"] = max_tokens
                else:
                    base_kwargs["max_tokens"] = max_tokens

                if self.reasoning_effort is not None:
                    try:
                        return use_client.chat.completions.create(
                            **base_kwargs,
                            reasoning_effort=self.reasoning_effort,
                        )
                    except Exception as e:
                        # Some providers/models don't support reasoning_effort.
                        if "reasoning_effort" in str(e):
                            return use_client.chat.completions.create(**base_kwargs)
                        raise

                return use_client.chat.completions.create(**base_kwargs)
            except Exception as e:
                retryable = isinstance(e, (RateLimitError, APIConnectionError, APITimeoutError, APIError))
                message = str(e).lower()
                if not retryable and not any(k in message for k in ["rate limit", "timeout", "connection", "temporarily", "429"]):
                    raise
                last_error = e
                if attempt >= self.max_api_retries:
                    break
                backoff = min(
                    self.initial_backoff * (self.backoff_multiplier ** attempt),
                    self.max_backoff,
                )
                print(f"  [RETRY] LLM call failed ({type(e).__name__}), retry in {backoff:.1f}s...")
                time.sleep(backoff)

            raise last_error
        
    def tokenize(self, sentence):
        """
        Tokenize sentence with MeCab when available.
        Falls back to character-level tokenization.
        """
        morphemes = self._analyze_morphemes(sentence)
        if morphemes:
            return [m.get("surface", "") for m in morphemes if m.get("surface")]
        return list(sentence)

    def tokenize_with_pos(self, sentence):
        """Step-2 output: token list enriched with POS/reading/span fields."""
        morphemes = self._analyze_morphemes(sentence)
        if morphemes:
            return [
                {
                    "surface": m.get("surface", ""),
                    "pos": m.get("pos", ""),
                    "reading": m.get("reading", ""),
                    "start": m.get("start", -1),
                    "end": m.get("end", -1),
                }
                for m in morphemes
                if m.get("surface")
            ]
        return [
            {
                "surface": ch,
                "pos": "",
                "reading": "",
                "start": idx,
                "end": idx + 1,
            }
            for idx, ch in enumerate(sentence or "")
        ]
    
    def lookup_dictionary(self, sentence):
        """
        Step 3: Lookup Homophone Dictionary
        Returns: [(word, position, candidates, contexts)]
        """
        return self._find_homophones(sentence)
    
    def generate_candidates_with_context(self, sentence, word_info):
        """
        Step 4: Generate candidates with IMPROVED context analysis
        Returns: (candidates_list, best_from_dict, confidence)
        """
        word = word_info["word"]
        candidates = word_info["candidates"]
        contexts = word_info["contexts"]
        
        # IMPROVED: Try dictionary-based selection with FLEXIBLE threshold
        best_candidate, confidence = self._get_best_candidate(
            word, sentence, candidates, contexts
        )
        
        # FALLBACK: If no context match, use safe lexical pattern rules
        if not best_candidate and candidates:
            for candidate in candidates:
                if candidate != word and candidate in sentence:
                    best_candidate = candidate
                    confidence = 2
                    break
        
        # Prepare candidates for LLM
        all_candidates = [word] + candidates
        
        return all_candidates, best_candidate, confidence
    
    def llm_select_best(self, sentence, word, candidates):
        """
        Step 5: AI recheck among dictionary candidates (and optional keep-original).
        """
        if len(candidates) <= 1:
            return word
        if self._llm_unavailable:
            return word
        
        correction_options = [c for c in candidates if c != word]
        options = correction_options if self.disable_keep_choice else [word] + correction_options + ["KEEP_ORIGINAL"]

        if not correction_options:
            return word

        candidates_str = ", ".join(options)
        embedding_scores = self._embedding_delta_scores(sentence, word, correction_options, client=self.client)
        embedding_hint = ""
        if embedding_scores:
            ranked = sorted(embedding_scores.items(), key=lambda x: x[1], reverse=True)
            top = [f"{cand}:{score:+.3f}" for cand, score in ranked[:3]]
            embedding_hint = f"Embedding候補優先度: {', '.join(top)}"
        
        # Removed external few-shot logger/formatter dependency.
        few_shot = ""
        
        output_format = f"""【出力形式】
    CHOICE: <候補>
    候補は次の中から1つのみ: {candidates_str}
    説明は絶対禁止"""

        if self.fast_llm_prompt:
            prompt = f"""同音異義語修正。
文: {sentence}
対象語: {word}
選択肢: {candidates_str}
{embedding_hint}

ルール:
- 候補から1つだけ出力（必ず CHOICE: 形式）
- 前後3〜5語の連語（collocation）を優先
- 単語意味の整合性を優先（文脈ベース）
- 言い換え禁止
{'- KEEP_ORIGINAL は全候補が不自然な時のみ' if not self.disable_keep_choice else '- KEEP_ORIGINAL は禁止'}
"""
        else:
            prompt = f"""【同音異義語誤り修正タスク】

文: {sentence}
対象語: {word}
選択肢: {candidates_str}
{embedding_hint}

【厳守ルール】
1. 選択肢以外の語を絶対に出力しない
2. 言い換え・同義語置換を絶対にしない
3. 文全体を書き直さない（1語だけ選ぶ）
4. {'KEEP_ORIGINAL は最終手段。候補の中に文脈上より自然な語があれば必ずその候補を選ぶ' if not self.disable_keep_choice else 'KEEP_ORIGINAL は禁止。必ず候補の中から選ぶ'}
5. 【最優先事項】意味が似ている別の単語への入れ替え（例: 意向→意見）はASR誤りでない限り厳禁
6. 対象語の前後3〜5語の連語自然性を最優先する
7. 文脈の意味整合を優先し、単純な文字一致より自然な候補を選ぶ
8. {'同点で迷う場合のみ KEEP_ORIGINAL を許可する' if not self.disable_keep_choice else '同点でも必ず候補を選ぶ'}

【Confusing Pairs Guidance】
- 追求: 目標・利益・理想を求める
- 追及: 責任・不正・犯行を問い詰める
- 追究: 学理・原因・本質を深く調べる

{few_shot}

{output_format}

判断:"""

        try:
            response = self._call_llm_with_retry(prompt, max_tokens=self.llm_select_max_completion_tokens)
            
            result = response.choices[0].message.content.strip()
            
            # Extract tokens used
            tokens_used = None
            if hasattr(response.usage, 'prompt_tokens'):
                tokens_used = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            
            # Parse CHOICE format
            normalized = result.replace("CHOICE:", "").replace("「", "").replace("」", "").replace("'", "").replace('"', "").strip()

            if normalized == "KEEP_ORIGINAL" and not self.disable_keep_choice:
                normalized = word
            if normalized == "KEEP_ORIGINAL" and self.disable_keep_choice:
                normalized = ""
            
            # Match against options
            final_choice = correction_options[0] if self.disable_keep_choice else word
            if normalized in options:
                final_choice = normalized
            elif normalized == "KEEP_ORIGINAL" and not self.disable_keep_choice:
                final_choice = word
            else:
                # Fallback: contains check
                for candidate in options:
                    if candidate in result:
                        final_choice = candidate
                        break
            
            return final_choice
            
        except Exception as e:
            self._mark_llm_unavailable_if_rate_limited(e)
            print(f"LLM error: {e}")
            if self.disable_keep_choice and correction_options:
                return correction_options[0]
            return word
    
    def correct_sentence(self, sentence):
        self.total_checked += 1
        self._last_used_llm = False
        self._last_llm_pairs = []
        self._last_no_dict_pairs = []

        def _finalize_output(text, change_list):
            final_text, post_changes = self.postprocess_spoken_style(text)
            final_changes = list(change_list)
            if post_changes:
                preview = "; ".join(post_changes[:2])
                if len(post_changes) > 2:
                    preview += "; ..."
                final_changes.append(f"POST[{preview}]")
            return final_text, final_changes

        pre_corrected, pre_changes = self._apply_verified_runtime_map(sentence)
        if pre_changes:
            sentence = pre_corrected
            print(f"  [MEM]  {'; '.join(pre_changes)}")

        _ = self.tokenize_with_pos(sentence)

        changes = list(pre_changes)
        corrected = sentence
        dict_applied_in_sentence = False
        llm_applied_in_sentence = False

        reverse_runtime_pairs = set()
        for item in pre_changes:
            if "→" not in item:
                continue
            left, right = item.split("→", 1)
            old = (left or "").strip()
            new = (right or "").strip()
            if old and new and old != new:
                # Block replacing corrected token back to original wrong token.
                reverse_runtime_pairs.add((new, old))

        homophone_matches = [] if self.disable_dictionary_stage else self.lookup_dictionary(corrected)
        homophone_matches_sorted = sorted(homophone_matches, key=lambda x: x["position"], reverse=True)

        for match in homophone_matches_sorted:
            word = match["word"]
            pos = match["position"]
            if self.skip_filler_in_dict_stage and word in self.llm_skip_tokens:
                continue
            original_candidates = match.get("candidates", [])
            if not original_candidates:
                continue

            candidate_pos_hints = {}
            for cand, rule in (match.get("candidate_rules", {}) or {}).items():
                if isinstance(rule, dict):
                    candidate_pos_hints[cand] = list(rule.get("pos_tags", []) or [])

            pos_filtered_candidates = self._filter_candidates_by_pos(
                corrected,
                word,
                pos,
                original_candidates,
                candidate_pos_hints=candidate_pos_hints,
            )
            if not pos_filtered_candidates:
                continue

            pos_filtered_match = dict(match)
            pos_filtered_match["candidates"] = pos_filtered_candidates
            candidates, best_dict, confidence = self.generate_candidates_with_context(corrected, pos_filtered_match)

            if not self._is_token_boundary_match(corrected, pos, len(word)):
                self._log_pos_prune(corrected, word, best_dict or "", "dict_span_not_on_token_boundary")
                continue

            def _can_apply(candidate):
                if not candidate or candidate == word:
                    return False
                candidate = self._sanitize_dictionary_token(candidate)
                if not candidate or candidate == word:
                    return False
                if self._is_blacklisted_mapping(word, candidate):
                    self._log_pos_prune(corrected, word, candidate, "blacklist_block")
                    return False
                norm_word = re.sub(r"[\s、。,.!?！？；;:：]+$", "", word or "")
                norm_candidate = re.sub(r"[\s、。,.!?！？；;:：]+$", "", candidate or "")
                if norm_word and norm_candidate and norm_word == norm_candidate:
                    self._log_pos_prune(corrected, word, candidate, "punctuation_only_block")
                    return False
                if (word, candidate) in reverse_runtime_pairs:
                    self._log_pos_prune(corrected, word, candidate, "runtime_reverse_block")
                    return False
                # Avoid injecting latin token into Japanese span unless source already has latin chars.
                if re.search(r"[A-Za-z]", candidate) and not re.search(r"[A-Za-z]", word):
                    self._log_pos_prune(corrected, word, candidate, "latin_noise_block")
                    return False
                if self._is_inside_protected_compound(corrected, pos, len(word)):
                    self._log_pos_prune(corrected, word, candidate, "protected_compound_block")
                    return False
                if self._is_protected_word(word):
                    self._log_pos_prune(corrected, word, candidate, "protected_word_block")
                    return False
                if self.no_dict_avoid_short_words and (
                    self._is_short_kana_fragment(word) or self._is_short_kana_fragment(candidate)
                ):
                    self._log_pos_prune(corrected, word, candidate, "short_kana_block")
                    return False
                if not self._passes_dict_rewrite_guard(corrected, word, candidate):
                    self._log_pos_prune(corrected, word, candidate, "dict_pos_guard_block")
                    return False
                return True

            effective_dict_threshold = self.dict_confidence_threshold
            if self._llm_unavailable:
                effective_dict_threshold = max(1, self.dict_confidence_threshold - 4)

            if best_dict and best_dict != word and confidence >= effective_dict_threshold:
                chosen = best_dict
                if (
                    self.llm_recheck_dict
                    and word not in self.llm_skip_tokens
                    and confidence <= (self.dict_confidence_threshold + self.dict_recheck_margin)
                ):
                    action, candidate = self._llm_recheck_dict_replacement(corrected, word, best_dict, candidates)
                    if action == "KEEP":
                        if self.disable_keep_choice and self.force_change_on_llm_uncertain:
                            chosen = candidate if candidate and candidate in candidates and candidate != word else best_dict
                        else:
                            self.llm_recheck_keep_override += 1
                            print(f"  [LLM-RECHECK] {word} kept")
                            continue
                    if action == "ALT" and candidate and candidate in candidates and candidate != word:
                        chosen = candidate
                        self.llm_recheck_alt_accept += 1
                    else:
                        self.llm_recheck_dict_accept += 1

                if not _can_apply(chosen):
                    continue

                corrected = corrected[:pos] + chosen + corrected[pos + len(word):]
                changes.append(f"{word}→{chosen}")
                if chosen == best_dict:
                    dict_applied_in_sentence = True
                    print(f"  [DICT] {word} → {chosen} (confidence: {confidence})")
                else:
                    llm_applied_in_sentence = True
                    self._last_used_llm = True
                    self._last_llm_pairs.append((word, chosen))
                    print(f"  [LLM-ALT] {word} → {chosen}")
                continue

            # Optional LLM select only for uncertain dict scores.
            if (
                self.enable_llm_select
                and len(candidates) > 1
                and not self._llm_unavailable
                and word not in self.llm_skip_tokens
                and confidence < effective_dict_threshold
            ):
                llm_choice = self.llm_select_best(corrected, word, candidates)
                if llm_choice == word:
                    if self.disable_keep_choice and self.force_change_on_llm_uncertain and best_dict and best_dict != word:
                        llm_choice = best_dict
                    else:
                        print(f"  [LLM-] {word} kept")
                        continue
                if llm_choice not in candidates:
                    print(f"  [LLM?] {word} -> {llm_choice} (invalid)")
                    continue
                if not _can_apply(llm_choice):
                    continue

                corrected = corrected[:pos] + llm_choice + corrected[pos + len(word):]
                changes.append(f"{word}→{llm_choice}")
                llm_applied_in_sentence = True
                self._last_used_llm = True
                self._last_llm_pairs.append((word, llm_choice))
                print(f"  [LLM]  {word} → {llm_choice}")

        # If unresolved after dictionary + AI recheck, route to no-dict LLM only
        # when detector/pos-anomaly reports a concrete target.
        if len(changes) == len(pre_changes) and self.no_dict_route_unresolved:
            should_route = False
            target_hint = ""
            
            if self.enable_no_dict_detector:
                should_route, target_hint = self._detect_no_dict_error(corrected)
            
            if not should_route and self.prioritize_pos_anomaly:
                anomaly_target = self._detect_pos_anomaly_target(corrected)
                if anomaly_target:
                    should_route = True
                    target_hint = anomaly_target
            if not should_route:
                final_text, final_changes = _finalize_output(corrected, pre_changes)
                if len(final_changes) == len(pre_changes):
                    self.no_change += 1
                return final_text, final_changes

            no_dict_input = corrected
            normalized_sentence, pre_norm_changes = self._normalize_pre_no_dict_sentence(no_dict_input)
            if normalized_sentence != no_dict_input:
                no_dict_input = normalized_sentence
                if pre_norm_changes:
                    preview = "; ".join(pre_norm_changes[:2])
                    if len(pre_norm_changes) > 2:
                        preview += "; ..."
                    changes.append(f"PRE[{preview}]")

            self.no_dict_routed += 1
            llm_corrected_sentence = self.llm_correct_without_dictionary(no_dict_input, target_hint=target_hint)

            if llm_corrected_sentence != no_dict_input:
                old_fragment, new_fragment = self._extract_single_span_change(no_dict_input, llm_corrected_sentence)
                if old_fragment and new_fragment:
                    self.llm_corrected += 1
                    self.no_dict_llm_changed += 1
                    self._last_used_llm = True
                    self._last_llm_pairs.append((old_fragment, new_fragment))
                    self._last_no_dict_pairs.append((old_fragment, new_fragment))
                    self._maybe_auto_learn_from_no_dict(old_fragment, new_fragment)
                    print(f"  [NO-DICT] {old_fragment} → {new_fragment}")
                    return _finalize_output(llm_corrected_sentence, changes + [f"{old_fragment}→{new_fragment}"])

            self.no_dict_llm_kept += 1

        if len(changes) == len(pre_changes):
            final_text, final_changes = _finalize_output(corrected, pre_changes)
            if len(final_changes) == len(pre_changes):
                self.no_change += 1
            return final_text, final_changes

        if dict_applied_in_sentence:
            self.dict_corrected += 1
        if llm_applied_in_sentence:
            self.llm_corrected += 1

        return _finalize_output(corrected, changes)
    
    def print_stats(self):
        """Print correction statistics"""
        print("\n" + "=" * 60)
        print("PIPELINE STATISTICS")
        print("=" * 60)
        print(f"Total sentences:      {self.total_checked}")
        print(f"Dict corrections:     {self.dict_corrected}")
        print(f"LLM corrections:      {self.llm_corrected}")
        print(f"No change needed:     {self.no_change}")
        print(f"No-dict routed:       {self.no_dict_routed}")
        print(f"No-dict LLM calls:    {self.no_dict_llm_calls}")
        print(f"No-dict changed:      {self.no_dict_llm_changed}")
        print(f"No-dict kept:         {self.no_dict_llm_kept}")
        print(f"No-dict rejected:     {self.no_dict_llm_rejected}")
        print(f"No-dict pair verify:  {self.no_dict_pair_verified_accept}")
        print(f"No-dict auto-learned: {self.no_dict_auto_learned}")
        print(f"No-dict short rejects:{self.no_dict_short_rejects}")
        print(f"POS candidates pruned:{self.pos_filtered_candidates}")
        print(f"POS prune events:     {self.pos_prune_events}")
        print(f"POS guard rejects:    {self.pos_guard_rejects}")
        print(f"LLM recheck calls:    {self.llm_recheck_calls}")
        print(f"LLM recheck DICT:     {self.llm_recheck_dict_accept}")
        print(f"LLM recheck KEEP:     {self.llm_recheck_keep_override}")
        print(f"LLM recheck ALT ok:   {self.llm_recheck_alt_accept}")
        print(f"Embedding hints used: {self.embedding_recheck_hints}")
        print(f"Runtime map learned:  {self.runtime_map_learned}")
        print(f"Runtime map applied:  {self.runtime_map_applied}")
        print(f"Post-normalized sent.:{self.post_normalized_sentences}")
        sentence_corrections = max(0, self.total_checked - self.no_change)
        print(f"Total corrections:    {sentence_corrections}")
        print("=" * 60)
        
