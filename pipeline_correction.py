#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

try:
    from groq import Groq
except ImportError:
    Groq = None

from openai import OpenAI

try:
    import MeCab
except ImportError:
    MeCab = None

try:
    from pykakasi import kakasi as pykakasi_factory
except ImportError:
    pykakasi_factory = None

try:
    from tools.japanese_mecab_helper import JapaneseMorphAnalyzer
except Exception:
    JapaneseMorphAnalyzer = None


class JapaneseNLP:
    """Handle MeCab tokenization/POS and reading similarity."""

    def __init__(self, dicdir: str = None, enable_pos: bool = True):
        self.enable_pos = enable_pos
        self.tagger = None
        self.yomi_tagger = None
        self._kakasi_converter = None
        self._reading_cache: Dict[str, str] = {}
        self._morph = None

        if pykakasi_factory is not None:
            try:
                kks = pykakasi_factory()
                # Backward-compatible init for old/new pykakasi APIs.
                if hasattr(kks, "setMode") and hasattr(kks, "getConverter"):
                    kks.setMode("J", "H")
                    kks.setMode("K", "H")
                    kks.setMode("H", "H")
                    self._kakasi_converter = kks.getConverter()
                else:
                    self._kakasi_converter = kks
                print("  [NLP] pykakasi initialized.")
            except Exception as e:
                print(f"  [NLP-WARNING] pykakasi init failed: {e}")
                self._kakasi_converter = None

        if self.enable_pos and JapaneseMorphAnalyzer is not None:
            try:
                self._morph = JapaneseMorphAnalyzer(dicdir=dicdir)
                if not getattr(self._morph, "available", False):
                    self._morph = None
                else:
                    print("  [NLP] JapaneseMorphAnalyzer initialized.")
            except Exception as e:
                print(f"  [NLP-WARNING] JapaneseMorphAnalyzer init failed: {e}")
                self._morph = None

        if self.enable_pos and MeCab is not None:
            try:
                self.tagger = MeCab.Tagger(f"-d {dicdir}" if dicdir else "")
                self.tagger.parse("")
                self.yomi_tagger = MeCab.Tagger(f"-d {dicdir} -Oyomi" if dicdir else "-Oyomi")
                self.yomi_tagger.parse("")
                print("  [NLP] MeCab initialized.")
            except Exception as e:
                print(f"  [NLP-WARNING] MeCab init failed: {e}")
                self.tagger = None
                self.yomi_tagger = None

    @staticmethod
    def _to_hiragana(text: str) -> str:
        return "".join(
            chr(ord(c) - 0x60) if 0x30A1 <= ord(c) <= 0x30F6 else c
            for c in (text or "")
        )

    def _reading_from_pykakasi(self, text: str) -> str:
        if not text or self._kakasi_converter is None:
            return ""
        try:
            # New API: kakasi().convert(...); old API: converter.do(...)
            if hasattr(self._kakasi_converter, "convert"):
                converted = self._kakasi_converter.convert(text)
                reading = "".join(
                    (item.get("hira") or item.get("kana") or item.get("orig") or "")
                    for item in converted
                    if isinstance(item, dict)
                )
            else:
                reading = self._kakasi_converter.do(text)
        except Exception:
            return ""
        cleaned = re.sub(r"[^\u3040-\u309f\u30a0-\u30ffー]", "", reading or "")
        return self._to_hiragana(cleaned)

    def get_reading(self, text: str) -> str:
        if not text:
            return ""
        if text in self._reading_cache:
            return self._reading_cache[text]

        # 1) Try pykakasi
        pykakasi_reading = self._reading_from_pykakasi(text)
        if pykakasi_reading:
            self._reading_cache[text] = pykakasi_reading
            return pykakasi_reading

        # 2) Try yomi tagger
        if self.yomi_tagger is not None:
            try:
                yomi = self.yomi_tagger.parse(text).splitlines()[0].strip()
                cleaned = self._to_hiragana(re.sub(r"[^\u3040-\u309f\u30a0-\u30ffー]", "", yomi))
                if cleaned:
                    self._reading_cache[text] = cleaned
                    return cleaned
            except Exception:
                pass

        # 3) Try morph analyzer tokens
        if self._morph is not None:
            try:
                toks = self._morph.tokenize(text)
                reading = "".join((t.get("reading") or t.get("surface") or "") for t in toks)
                cleaned = self._to_hiragana(re.sub(r"[^\u3040-\u309f\u30a0-\u30ffー]", "", reading))
                if cleaned:
                    self._reading_cache[text] = cleaned
                    return cleaned
            except Exception:
                pass

        self._reading_cache[text] = text
        return text

    def reading_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        r1 = self._to_hiragana(self.get_reading(text1))
        r2 = self._to_hiragana(self.get_reading(text2))
        s1 = self._to_hiragana(text1)
        s2 = self._to_hiragana(text2)
        return max(SequenceMatcher(None, r1, r2).ratio(), SequenceMatcher(None, s1, s2).ratio())

    def get_morphemes(self, text: str) -> List[dict]:
        if not text:
            return []

        # 1) Preferred path: helper analyzer
        if self._morph is not None:
            try:
                raw_tokens = self._morph.tokenize(text)
                tokens = []
                cursor = 0
                for t in raw_tokens:
                    surface = t.get("surface") or ""
                    if not surface:
                        continue
                    start = text.find(surface, cursor)
                    if start < 0:
                        start = cursor
                    end = start + len(surface)
                    tokens.append(
                        {
                            "surface": surface,
                            "start": start,
                            "end": end,
                            "pos": t.get("pos", ""),
                            "base": t.get("base", surface),
                            "reading": t.get("reading", ""),
                        }
                    )
                    cursor = end
                if tokens:
                    return tokens
            except Exception:
                pass

        # 2) Fallback path: direct MeCab
        if self.tagger is not None:
            try:
                tokens = []
                cursor = 0
                node = self.tagger.parseToNode(text)
                while node:
                    surface = node.surface or ""
                    if surface:
                        start = text.find(surface, cursor)
                        if start < 0:
                            start = cursor
                        end = start + len(surface)
                        feats = (node.feature or "").split(",")
                        tokens.append(
                            {
                                "surface": surface,
                                "start": start,
                                "end": end,
                                "pos": feats[0] if len(feats) > 0 else "",
                                "base": feats[6] if len(feats) > 6 and feats[6] != "*" else surface,
                                "reading": feats[7] if len(feats) > 7 and feats[7] != "*" else "",
                            }
                        )
                        cursor = end
                    node = node.next
                if tokens:
                    return tokens
            except Exception:
                pass

        # 3) Last resort: character tokens
        return [
            {
                "surface": ch,
                "start": i,
                "end": i + 1,
                "pos": "",
                "base": ch,
                "reading": ch,
            }
            for i, ch in enumerate(text)
            if ch.strip()
        ]

    def is_token_boundary(self, text: str, start: int, length: int) -> bool:
        toks = self.get_morphemes(text)
        if not toks:
            return True
        end = start + length
        has_start = any(t["start"] == start for t in toks)
        has_end = any(t["end"] == end for t in toks)
        return has_start and has_end


class ASRLLMEngine:
    _throttle_lock = threading.Lock()
    _next_allowed_by_key: Dict[str, float] = {}
    _semaphore_lock = threading.Lock()
    _semaphore_pool: Dict[str, threading.Semaphore] = {}

    def __init__(self, config: dict):
        self.config = config
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "")
        self.model = config.get("model", "")
        self.request_timeout = float(config.get("request_timeout", 120))
        self.max_retries = int(config.get("max_api_retries", 3))
        self.min_request_interval_sec = max(0.0, float(config.get("min_request_interval_sec", 0.0) or 0.0))
        self.max_concurrent_requests = max(1, int(config.get("max_concurrent_requests", 1) or 1))
        self.safe_max_tokens_cap = max(0, int(config.get("safe_max_tokens_cap", 0) or 0))
        self._throttle_key = f"{self.base_url}|{self.model}"
        self._request_semaphore = self._get_or_create_semaphore()
        self.client = self._init_client()
        self.unavailable = False

        self.forbidden_pairs = self._parse_forbidden(config.get("zero_shot_forbidden_replacements", []))

    def _init_client(self):
        if self.config.get("use_groq_sdk", True) and Groq and "api.groq.com" in self.base_url:
            return Groq(api_key=self.api_key)
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,
            timeout=self.request_timeout,
        )

    def _get_or_create_semaphore(self) -> threading.Semaphore:
        key = f"{self.base_url}|{self.max_concurrent_requests}"
        with self.__class__._semaphore_lock:
            sem = self.__class__._semaphore_pool.get(key)
            if sem is None:
                sem = threading.Semaphore(self.max_concurrent_requests)
                self.__class__._semaphore_pool[key] = sem
            return sem

    def _cap_tokens(self, max_tokens: int) -> int:
        capped = max(1, int(max_tokens))
        if self.safe_max_tokens_cap > 0:
            capped = min(capped, self.safe_max_tokens_cap)
        return capped

    def _acquire_request_slot(self) -> None:
        self._request_semaphore.acquire()
        if self.min_request_interval_sec <= 0.0:
            return

        while True:
            with self.__class__._throttle_lock:
                now = time.monotonic()
                next_allowed = self.__class__._next_allowed_by_key.get(self._throttle_key, 0.0)
                wait_sec = next_allowed - now
                if wait_sec <= 0.0:
                    self.__class__._next_allowed_by_key[self._throttle_key] = now + self.min_request_interval_sec
                    return
            time.sleep(min(wait_sec, 0.2))

    def _release_request_slot(self) -> None:
        self._request_semaphore.release()

    @staticmethod
    def _strip_reasoning(raw_text: str) -> str:
        text = raw_text or ""
        # Remove common reasoning tags emitted by reasoning-capable models.
        text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<analysis\b[^>]*>.*?</analysis>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?analysis\b[^>]*>", "", text, flags=re.IGNORECASE)
        # Handle malformed output where a reasoning tag opens but is never closed.
        if re.search(r"<think\b|<analysis\b", text, flags=re.IGNORECASE):
            text = re.split(r"<think\b|<analysis\b", text, flags=re.IGNORECASE)[0]
        text = re.sub(r"^```[\\w-]*\\s*", "", text.strip())
        text = re.sub(r"\\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _parse_forbidden(raw_list: List[str]) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()
        for item in raw_list:
            if "->" in item:
                a, b = item.split("->", 1)
                pairs.add((a.strip(), b.strip()))
        return pairs

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        if self.unavailable:
            return ""

        tmp = self.config.get("temperature", 0.0) if temperature is None else temperature
        tp = self.config.get("top_p", 0.4) if top_p is None else top_p
        req_tokens = self._cap_tokens(max_tokens)

        for attempt in range(self.max_retries + 1):
            acquired_slot = False
            try:
                self._acquire_request_slot()
                acquired_slot = True
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": tmp,
                    "top_p": tp,
                }
                if isinstance(self.client, Groq):
                    kwargs["max_completion_tokens"] = req_tokens
                else:
                    kwargs["max_tokens"] = req_tokens

                response = self.client.chat.completions.create(**kwargs)
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                err = str(e).lower()
                if ("rate limit" in err or "429" in err) and self.config.get("disable_llm_on_rate_limit", True):
                    print("  [LLM] Rate limited. Disable further calls in this run.")
                    self.unavailable = True
                    return ""
                if attempt < self.max_retries:
                    sleep_t = min(float(self.config.get("max_backoff", 60.0)), float(self.config.get("initial_backoff", 2.0)) * (2 ** attempt))
                    print(f"  [LLM-RETRY] {type(e).__name__}: retry in {sleep_t:.1f}s")
                    time.sleep(sleep_t)
                else:
                    print(f"  [LLM-ERROR] {e}")
            finally:
                if acquired_slot:
                    self._release_request_slot()
        return ""

    def generate_completion(
        self,
        prompt: str,
        max_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        if self.unavailable:
            return ""

        tmp = self.config.get("temperature", 0.0) if temperature is None else temperature
        tp = self.config.get("top_p", 0.4) if top_p is None else top_p
        req_tokens = self._cap_tokens(max_tokens)

        for attempt in range(self.max_retries + 1):
            acquired_slot = False
            try:
                self._acquire_request_slot()
                acquired_slot = True
                # Mirror standalone eval script behavior: prompt template + completions API.
                if isinstance(self.client, Groq):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=req_tokens,
                        temperature=tmp,
                        top_p=tp,
                    )
                    text = (response.choices[0].message.content or "").strip()
                    return text

                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    stream=False,
                    max_tokens=req_tokens,
                    temperature=tmp,
                    top_p=tp,
                    timeout=self.request_timeout,
                )
                if not response or not getattr(response, "choices", None):
                    return ""
                choice0 = response.choices[0]
                text = getattr(choice0, "text", None)
                if text is None and getattr(choice0, "message", None) is not None:
                    text = getattr(choice0.message, "content", "")
                return (text or "").strip()
            except Exception as e:
                err = str(e).lower()
                if ("rate limit" in err or "429" in err) and self.config.get("disable_llm_on_rate_limit", True):
                    print("  [LLM] Rate limited. Disable further calls in this run.")
                    self.unavailable = True
                    return ""
                if attempt < self.max_retries:
                    sleep_t = min(float(self.config.get("max_backoff", 60.0)), float(self.config.get("initial_backoff", 2.0)) * (2 ** attempt))
                    print(f"  [LLM-RETRY] {type(e).__name__}: retry in {sleep_t:.1f}s")
                    time.sleep(sleep_t)
                else:
                    print(f"  [LLM-ERROR] {e}")
            finally:
                if acquired_slot:
                    self._release_request_slot()
        return ""

    def parse_choice(self, raw_output: str, candidates: List[str]) -> str:
        text = self._strip_reasoning(raw_output)
        if not text:
            return "KEEP"

        if "<選択>" in text:
            m = re.search(r"<選択>\[(.*?)\]", text)
            if m:
                text = m.group(1).strip()

        upper = text.upper()
        if "KEEP" in upper:
            return "KEEP"

        for cand in candidates:
            if cand and cand in text:
                return cand

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            first = lines[0]
            for cand in candidates:
                if first == cand:
                    return cand

        return "KEEP"

    def parse_correction(self, raw_output: str, original_text: str) -> str:
        if not raw_output:
            return original_text

        text = self._strip_reasoning(raw_output)

        m = re.search(r"<改>\[(.*?)\]", text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()

        text = re.sub(r"^出力\s*[:：]\s*", "", text).strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return original_text

        # pick first meaningful sentence-like line to avoid model explanations
        for ln in lines:
            if not re.match(r"^(入力|input)\b", ln, flags=re.IGNORECASE):
                return ln
        return original_text

    @staticmethod
    def extract_span(original: str, corrected: str) -> Tuple[str, str]:
        if original == corrected:
            return "", ""

        prefix = 0
        min_len = min(len(original), len(corrected))
        while prefix < min_len and original[prefix] == corrected[prefix]:
            prefix += 1

        suffix = 0
        o_rem = len(original) - prefix
        c_rem = len(corrected) - prefix
        min_rem = min(o_rem, c_rem)
        while suffix < min_rem and original[-(suffix + 1)] == corrected[-(suffix + 1)]:
            suffix += 1

        old_frag = original[prefix : len(original) - suffix if suffix else len(original)]
        new_frag = corrected[prefix : len(corrected) - suffix if suffix else len(corrected)]
        return old_frag, new_frag

    def passes_postcheck(self, original: str, corrected: str, old_frag: str, new_frag: str) -> bool:
        if not old_frag or not new_frag:
            return False
        if abs(len(new_frag) - len(old_frag)) > int(self.config.get("zero_shot_max_char_delta", 18)):
            return False
        if (old_frag, new_frag) in self.forbidden_pairs:
            return False
        if corrected.strip() == original.strip():
            return False
        return True


class PipelineCorrector:
    """Pipeline: Tokenize/POS -> Dict scan -> POS filter -> scoring -> Rule or AI recheck -> No-dict fallback."""

    def __init__(self, config: dict):
        self.config = config or {}

        # Core modules
        self.nlp = JapaneseNLP(
            dicdir=self.config.get("mecab_dicdir"),
            enable_pos=bool(self.config.get("enable_pos_filter", True)),
        )
        self.llm = ASRLLMEngine(self.config)

        # Main switches
        self.llm_no_dict_only_mode = bool(self.config.get("llm_no_dict_only_mode", False))
        self.zero_shot_translation_mode = bool(self.config.get("zero_shot_translation_mode", False))
        self.disable_dictionary_stage = bool(self.config.get("disable_dictionary_stage", False))
        self.enable_no_dict_detector = bool(self.config.get("enable_no_dict_detector", True))
        # Keep pipeline generic: do not depend on corpus-derived collocation priors.
        self.enable_collocation_score = False
        self.auto_learn_dictionary = bool(self.config.get("auto_learn_dictionary", True))
        self.disable_seed_replacements = bool(self.config.get("disable_seed_replacements", False))
        self.no_dict_run_on_residual_suspicious = bool(self.config.get("no_dict_run_on_residual_suspicious", True))

        if self.llm_no_dict_only_mode:
            self.zero_shot_translation_mode = True
            self.disable_dictionary_stage = True
            self.enable_no_dict_detector = True
            self.auto_learn_dictionary = False
            self.disable_seed_replacements = True

        # Scoring weights / thresholds
        self.rule_score_threshold = float(self.config.get("rule_score_threshold", 2.6))
        self.rule_score_margin = float(self.config.get("rule_score_margin", 0.35))
        self.keyword_score_weight = float(self.config.get("keyword_score_weight", 0.8))
        self.reading_similarity_weight = float(self.config.get("reading_similarity_weight", 1.1))
        self.metadata_weight = float(self.config.get("metadata_weight", 0.6))
        self.rule_min_reading_similarity = float(self.config.get("rule_min_reading_similarity", 0.45))
        self.recheck_max_candidates = int(self.config.get("recheck_max_candidates", 3))
        self.recheck_min_score_to_call = float(self.config.get("recheck_min_score_to_call", 1.8))

        # No-dict branch constraints
        self.no_dict_min_reading_similarity = float(self.config.get("no_dict_min_reading_similarity", 0.55))
        self.no_dict_sentence_max_edit_chars = int(self.config.get("no_dict_sentence_max_edit_chars", 12))

        # Safety / routing hints
        raw_forbidden = list(self.config.get("forbidden_replacements", []))
        raw_forbidden += list(self.config.get("zero_shot_forbidden_replacements", []))
        self.forbidden_replacements = self._parse_replacement_pairs(raw_forbidden)

        # Spoken-style normalization is intentionally disabled for general behavior.
        self.post_normalization_regex_rules = []

        # Dictionary and caches
        if self.llm_no_dict_only_mode:
            self.homophones_file = ""
            self.homophone_groups = {}
            self.homophones = {}
            self.no_dict_signal_terms, self.no_dict_signal_patterns = [], []
            self.seed_replacements = {}
        else:
            self.homophones_file = self.config.get("homophones_file", "data/japanese/dictionary/homophones.json")
            self.homophone_groups = self._load_homophone_groups(self.homophones_file)
            self.homophones = self._build_surface_homophone_map(self.homophone_groups)
            self.no_dict_signal_terms, self.no_dict_signal_patterns = self._load_no_dict_signals(self.homophone_groups)
            self.seed_replacements = self._load_seed_replacements(self.homophone_groups)

        self.zero_shot_cache: Dict[str, str] = {}
        self.verified_runtime_map: Dict[str, str] = {}
        self.runtime_blacklist: Set[Tuple[str, str]] = set()
        self._pending_learn_pairs: List[Tuple[str, str, str]] = []

        # Collocation index
        self.bigram_counts: Counter = Counter()
        self.trigram_counts: Counter = Counter()
        self.collocation_ready = False

        # Compatibility counters expected by runner
        self.total_checked = 0
        self.dict_corrected = 0
        self.llm_corrected = 0
        self.llm_suggested = 0
        self.no_change = 0
        self.no_dict_routed = 0
        self.no_dict_llm_calls = 0
        self.no_dict_llm_changed = 0
        self.no_dict_llm_kept = 0
        self.no_dict_llm_rejected = 0
        self.no_dict_short_rejects = 0
        self.no_dict_pair_verified_accept = 0
        self.no_dict_auto_learned = 0
        self.seed_map_applied = 0
        self.runtime_map_learned = 0
        self.runtime_map_applied = 0
        self.post_normalized_sentences = 0
        self.no_dict_pre_normalized_sentences = 0

        # Internal stats for debug
        self.stats = defaultdict(int)
        self.llm_recheck_calls = 0
        self.llm_recheck_kept = 0
        self.llm_recheck_changed = 0

        if not self.llm_no_dict_only_mode:
            self._load_caches()
            print(f"Loaded {len(self.homophone_groups)} reading groups from {self.homophones_file}")
        else:
            print("LLM no-dict-only mode: dictionary/caches disabled")

    # ------------------------------------------------------------------
    # Load / Save helpers
    # ------------------------------------------------------------------
    def _load_homophone_groups(self, file_path: str) -> Dict[str, dict]:
        if not file_path or not os.path.exists(file_path):
            return {}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"  [DICT-WARNING] Failed to load {file_path}: {e}")
            return {}

    @staticmethod
    def _parse_replacement_pairs(raw_list: List[str]) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()
        for item in raw_list:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if not text:
                continue
            if "->" in text:
                a, b = text.split("->", 1)
            elif "→" in text:
                a, b = text.split("→", 1)
            else:
                continue
            a = a.strip()
            b = b.strip()
            if a and b:
                pairs.add((a, b))
        return pairs

    @staticmethod
    def _load_no_dict_signals(groups: Dict[str, dict]) -> Tuple[List[str], List[re.Pattern]]:
        terms: List[str] = []
        patterns: List[re.Pattern] = []

        if not isinstance(groups, dict):
            return terms, patterns

        meta = groups.get("__pipeline_meta__", {})
        if not isinstance(meta, dict):
            return terms, patterns

        # Handle both old list format and new candidates format for no_dict_signal_terms
        raw_terms = meta.get("no_dict_signal_terms", [])
        if isinstance(raw_terms, list):
            for item in raw_terms:
                t = str(item or "").strip()
                if t:
                    terms.append(t)
        elif isinstance(raw_terms, dict):
            candidates = raw_terms.get("candidates", [])
            for cand in candidates:
                if isinstance(cand, dict):
                    word = cand.get("word", "")
                    t = str(word or "").strip()
                    if t:
                        terms.append(t)

        # Handle both old list format and new candidates format for no_dict_signal_patterns
        raw_patterns = meta.get("no_dict_signal_patterns", [])
        if isinstance(raw_patterns, list):
            for item in raw_patterns:
                p = str(item or "").strip()
                if not p:
                    continue
                try:
                    patterns.append(re.compile(p))
                except re.error:
                    continue
        elif isinstance(raw_patterns, dict):
            candidates = raw_patterns.get("candidates", [])
            for cand in candidates:
                if isinstance(cand, dict):
                    word = cand.get("word", "")
                    p = str(word or "").strip()
                    if not p:
                        continue
                    try:
                        patterns.append(re.compile(p))
                    except re.error:
                        continue

        return terms, patterns

    @staticmethod
    def _load_seed_replacements(groups: Dict[str, dict]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if not isinstance(groups, dict):
            return mapping

        meta = groups.get("__pipeline_meta__", {})
        if not isinstance(meta, dict):
            return mapping

        raw = meta.get("seed_replacements", {})
        
        # Handle old dict format: {"wrong": "correct", ...}
        if isinstance(raw, dict) and raw and "candidates" not in raw:
            for wrong, correct in raw.items():
                w = str(wrong or "").strip()
                c = str(correct or "").strip()
                if w and c and w != c:
                    mapping[w] = c
            return mapping

        # Handle new candidates format: {"candidates": [{word: "wrong", correct: "correct", ...}, ...]}
        if isinstance(raw, dict):
            candidates = raw.get("candidates", [])
            for cand in candidates:
                if isinstance(cand, dict):
                    word = cand.get("word", "")
                    correct = cand.get("correct", "")
                    w = str(word or "").strip()
                    c = str(correct or "").strip()
                    if w and c and w != c:
                        mapping[w] = c
            return mapping

        # Handle legacy list format with "->" or "→" separator
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, str):
                    continue
                text = item.strip()
                if "->" in text:
                    w, c = text.split("->", 1)
                elif "→" in text:
                    w, c = text.split("→", 1)
                else:
                    continue
                w = w.strip()
                c = c.strip()
                if w and c and w != c:
                    mapping[w] = c
        
        return mapping

    def _save_homophone_groups(self) -> None:
        if not self.homophones_file:
            return
        os.makedirs(os.path.dirname(self.homophones_file), exist_ok=True)
        with open(self.homophones_file, "w", encoding="utf-8") as f:
            json.dump(self.homophone_groups, f, ensure_ascii=False, indent=2)

    def _build_surface_homophone_map(self, groups: Dict[str, dict]) -> Dict[str, dict]:
        surface_map: Dict[str, dict] = {}
        if not isinstance(groups, dict):
            return surface_map

        for reading, group in groups.items():
            if isinstance(reading, str) and reading.startswith("__"):
                continue
            if not isinstance(group, dict):
                continue
            raw_candidates = group.get("candidates", [])
            candidate_items = []
            for cand in raw_candidates:
                if isinstance(cand, dict):
                    word = str(cand.get("word", "") or "").strip()
                    if not word:
                        continue
                    candidate_items.append((word, cand))
                else:
                    word = str(cand or "").strip()
                    if not word:
                        continue
                    candidate_items.append((word, {"word": word, "weight": 1.0, "context_rules": {}, "pos_tags": []}))

            words = [w for w, _ in candidate_items]
            for word, meta in candidate_items:
                others = [w for w in words if w != word]
                entry = surface_map.setdefault(word, {"reading": reading, "candidates": [], "candidate_meta": {}})
                for o in others:
                    if o not in entry["candidates"]:
                        entry["candidates"].append(o)
                for o_word, o_meta in candidate_items:
                    if o_word == word:
                        continue
                    entry["candidate_meta"][o_word] = o_meta

        return surface_map

    def _refresh_surface_map(self) -> None:
        self.homophones = self._build_surface_homophone_map(self.homophone_groups)

    def _load_caches(self) -> None:
        blacklist_path = self.config.get("runtime_map_blacklist_path")
        if blacklist_path and os.path.exists(blacklist_path):
            try:
                with open(blacklist_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        w = str(data.get("wrong_word", "") or "")
                        c = str(data.get("correct_word", "") or "")
                        if w and c:
                            self.runtime_blacklist.add((w, c))
            except Exception:
                pass

        runtime_path = self.config.get("verified_runtime_path")
        if runtime_path and os.path.exists(runtime_path):
            try:
                with open(runtime_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries = data.get("entries", {}) if isinstance(data, dict) else {}
                for wrong, info in entries.items():
                    if not isinstance(info, dict):
                        continue
                    correct = str(info.get("correct", "") or "")
                    if wrong and correct and (wrong, correct) not in self.runtime_blacklist:
                        self.verified_runtime_map[wrong] = correct
            except Exception:
                pass

    def _save_runtime_map(self) -> None:
        path = self.config.get("verified_runtime_path")
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "schema": "runtime_verified_map_v1",
            "generated_at": int(time.time()),
            "entries": {w: {"correct": c, "hits": 2} for w, c in self.verified_runtime_map.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _extract_line_text(line: str) -> str:
        line = (line or "").strip()
        if not line:
            return ""
        parts = line.split(" ", 1)
        if len(parts) == 2 and re.match(r"^[A-Za-z0-9_-]+$", parts[0]):
            return parts[1].strip()
        return line

    def _build_collocation_index(self, corpus_paths: List[str]) -> None:
        loaded = 0
        for p in corpus_paths:
            if not p or not os.path.exists(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        text = self._extract_line_text(line)
                        if not text:
                            continue
                        toks = [t["surface"] for t in self.nlp.get_morphemes(text)]
                        if len(toks) < 2:
                            continue
                        for i in range(len(toks) - 1):
                            self.bigram_counts[(toks[i], toks[i + 1])] += 1
                        for i in range(len(toks) - 2):
                            self.trigram_counts[(toks[i], toks[i + 1], toks[i + 2])] += 1
                        loaded += 1
            except Exception:
                continue

        self.collocation_ready = len(self.bigram_counts) > 0
        print(f"  [COLLOCATION] sentences={loaded}, bigrams={len(self.bigram_counts)}, trigrams={len(self.trigram_counts)}")

    @staticmethod
    def _char_ngram_vector(text: str, n: int = 2) -> Dict[str, float]:
        t = re.sub(r"\s+", "", text or "")
        if not t:
            return {}
        vec: Dict[str, float] = defaultdict(float)
        if len(t) < n:
            vec[t] += 1.0
            return vec
        for i in range(len(t) - n + 1):
            vec[t[i : i + n]] += 1.0
        return vec

    @staticmethod
    def _cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        if not v1 or not v2:
            return 0.0
        inter = set(v1.keys()) & set(v2.keys())
        dot = sum(v1[k] * v2[k] for k in inter)
        n1 = math.sqrt(sum(x * x for x in v1.values()))
        n2 = math.sqrt(sum(x * x for x in v2.values()))
        if n1 <= 0.0 or n2 <= 0.0:
            return 0.0
        return dot / (n1 * n2)

    def _embedding_context_similarity(self, candidate: str, left_ctx: str, right_ctx: str) -> float:
        # Lightweight embedding-like semantic signal from character n-gram vectors.
        context_text = f"{left_ctx} {right_ctx}".strip()
        v_cand = self._char_ngram_vector(candidate, n=2)
        v_ctx = self._char_ngram_vector(context_text, n=2)
        return self._cosine_similarity(v_cand, v_ctx)

    @staticmethod
    def _frequency_to_weight(freq_value) -> float:
        if freq_value is None:
            return 0.0
        if isinstance(freq_value, (int, float)):
            return float(freq_value)
        text = str(freq_value).strip().lower()
        mapping = {
            "very_high": 2.0,
            "high": 1.2,
            "medium": 0.7,
            "low": 0.3,
        }
        return mapping.get(text, 0.0)

    def _metadata_score(self, meta: dict) -> float:
        if not isinstance(meta, dict):
            return 0.0
        w = float(meta.get("weight", 1.0) or 1.0)
        freq = self._frequency_to_weight(meta.get("frequency", meta.get("freq")))
        score = self.metadata_weight * (0.6 * w + 0.4 * freq)

        # Penalize candidates without context hints to reduce rare/odd selections.
        rules = meta.get("context_rules", {}) if isinstance(meta.get("context_rules"), dict) else {}
        require_any = rules.get("require_any", [])
        require_coll = rules.get("require_collocation", [])
        if not require_any and not require_coll:
            score -= 0.35 * self.metadata_weight
        return score

    def _context_keyword_score(self, sentence: str, left_ctx: str, right_ctx: str, meta: dict) -> float:
        if not isinstance(meta, dict):
            return 0.0
        rules = meta.get("context_rules", {}) if isinstance(meta.get("context_rules"), dict) else {}
        require_any = rules.get("require_any", [])
        require_coll = rules.get("require_collocation", [])
        exclude = rules.get("exclude", [])

        context = f"{left_ctx} {right_ctx}"
        full = f"{sentence} {context}"

        hit_any = sum(1 for kw in require_any if kw and kw in full)
        hit_coll = sum(1 for kw in require_coll if kw and kw in sentence)
        hit_ex = sum(1 for kw in exclude if kw and kw in full)

        raw = (0.55 * hit_any) + (0.9 * hit_coll) - (1.2 * hit_ex)
        return self.keyword_score_weight * raw

    def _collocation_score(self, left_word: str, candidate: str, right_word: str) -> float:
        if not self.enable_collocation_score or not self.collocation_ready:
            return 0.0

        score = 0.0
        if left_word:
            score += math.log1p(self.bigram_counts[(left_word, candidate)])
        if right_word:
            score += math.log1p(self.bigram_counts[(candidate, right_word)])
        if left_word and right_word:
            score += 0.6 * math.log1p(self.trigram_counts[(left_word, candidate, right_word)])
        return score

    def _score_candidate(
        self,
        sentence: str,
        tokens: List[dict],
        idx: int,
        token: dict,
        candidate: str,
        meta: dict,
    ) -> Tuple[float, dict]:
        original = token.get("surface", "")
        token_pos = token.get("pos", "")

        if (original, candidate) in self.forbidden_replacements:
            return -9999.0, {"rejected": True, "reason": "forbidden"}

        pos_tags = meta.get("pos_tags", []) if isinstance(meta, dict) else []
        if self.config.get("enable_pos_filter", True) and pos_tags and token_pos:
            if not any(token_pos.startswith(p) for p in pos_tags):
                return -9999.0, {"rejected": True, "reason": "pos_filtered"}

        left_word = tokens[idx - 1]["surface"] if idx > 0 else ""
        right_word = tokens[idx + 1]["surface"] if idx + 1 < len(tokens) else ""
        left_ctx = "".join(t["surface"] for t in tokens[max(0, idx - 3) : idx])
        right_ctx = "".join(t["surface"] for t in tokens[idx + 1 : idx + 4])

        kw_score = self._context_keyword_score(sentence, left_ctx, right_ctx, meta)
        coll_score = 0.0
        read_sim = self.nlp.reading_similarity(original, candidate)
        emb_sim = self._embedding_context_similarity(candidate, left_ctx, right_ctx)
        meta_score = self._metadata_score(meta)

        total = kw_score
        total += self.reading_similarity_weight * read_sim
        total += meta_score
        total += 0.4 * emb_sim

        if read_sim < self.rule_min_reading_similarity:
            total -= (self.rule_min_reading_similarity - read_sim) * 2.5

        detail = {
            "kw_score": kw_score,
            "coll_score": coll_score,
            "reading_similarity": read_sim,
            "embedding_similarity": emb_sim,
            "meta_score": meta_score,
            "total": total,
            "rejected": False,
        }
        return total, detail

    def _has_residual_suspicion(self, sentence: str) -> bool:
        text = sentence or ""
        if not text:
            return False

        if not self.no_dict_signal_terms and not self.no_dict_signal_patterns:
            # Backward-compatible behavior: if no explicit signals configured, allow no-dict fallback.
            return True

        if any(term and term in text for term in self.no_dict_signal_terms):
            return True
        if any(p.search(text) for p in self.no_dict_signal_patterns):
            return True

        # Lightweight generic heuristics for unresolved ASR artifacts.
        if re.search(r"[A-Za-z]{3,}", text):
            return True
        if re.search(r"[一-龯ぁ-んァ-ヶー]{1,20}\.$", text):
            return True
        if re.search(r"(意向|型でも|下さ(?:い|った)|有る|無い|出来る|聞き手たい)", text):
            return True
        return False

    def _apply_seed_replacements(self, sentence: str) -> Tuple[str, List[str]]:
        if self.disable_seed_replacements:
            return sentence, []
        if not self.seed_replacements:
            return sentence, []

        updated = sentence
        changes: List[str] = []

        for wrong in sorted(self.seed_replacements.keys(), key=len, reverse=True):
            correct = self.seed_replacements[wrong]
            start = updated.find(wrong)
            if start < 0:
                continue
            if self.config.get("no_dict_token_boundary_check", True):
                if not self.nlp.is_token_boundary(updated, start, len(wrong)):
                    continue
            if (wrong, correct) in self.forbidden_replacements:
                continue
            updated = updated.replace(wrong, correct, 1)
            changes.append(f"{wrong}→{correct}")

        return updated, changes

    def _apply_memory_cache(self, sentence: str) -> Tuple[str, List[str]]:
        if not self.config.get("use_verified_runtime_map", True):
            return sentence, []

        updated = sentence
        changes: List[str] = []

        # Verified runtime map stores only feedback-confirmed replacements.
        for wrong in sorted(self.verified_runtime_map.keys(), key=len, reverse=True):
            correct = self.verified_runtime_map[wrong]
            start = updated.find(wrong)
            if start < 0:
                continue
            if self.config.get("no_dict_token_boundary_check", True):
                if not self.nlp.is_token_boundary(updated, start, len(wrong)):
                    continue
            updated = updated.replace(wrong, correct, 1)
            changes.append(f"{wrong}→{correct}")
        return updated, changes

    @staticmethod
    def _sanitize_generated_sentence(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""

        cleaned = re.sub(
            r"^(?:output|出力|修正文|しゅうせいぶん)\s*[:：]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # Remove enclosing quotes that models sometimes add around the full sentence.
        for _ in range(3):
            prev = cleaned
            if cleaned.startswith('"""') and cleaned.endswith('"""') and len(cleaned) >= 6:
                cleaned = cleaned[3:-3].strip()
            elif cleaned.startswith("'''") and cleaned.endswith("'''") and len(cleaned) >= 6:
                cleaned = cleaned[3:-3].strip()
            elif len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'", "`"}:
                cleaned = cleaned[1:-1].strip()
            if cleaned == prev:
                break

        cleaned = re.sub(r"[\"'`]{3,}\s*$", "", cleaned).strip()
        cleaned = re.sub(r"[\"'`]+\s*$", "", cleaned).strip()
        return cleaned

    @staticmethod
    def _count_edit_chars(source: str, target: str) -> int:
        edits = 0
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, source or "", target or "").get_opcodes():
            if tag == "equal":
                continue
            edits += max(i2 - i1, j2 - j1)
        return edits

    @staticmethod
    def _latin_tokens(text: str) -> Set[str]:
        return {tok.lower() for tok in re.findall(r"[A-Za-z]{2,}", text or "")}

    @staticmethod
    def _strip_soft_punctuation(text: str) -> str:
        return re.sub(r"[\s。．\.、,，！!？?・:：;；\-ー\"'`]+", "", text or "")

    def _has_unexpected_latin_token(self, original: str, corrected: str) -> bool:
        orig = self._latin_tokens(original)
        corr = self._latin_tokens(corrected)
        if not corr:
            return False
        return any(tok not in orig for tok in corr)

    def _validate_no_dict_candidate(
        self,
        original: str,
        corrected: str,
        old_frag: str,
        new_frag: str,
    ) -> Tuple[bool, str]:
        if not corrected or corrected == original:
            return False, "unchanged"

        if self._has_unexpected_latin_token(original, corrected):
            return False, "unexpected_latin_token"

        if not old_frag or not new_frag:
            return False, "empty_changed_span"

        if self._strip_soft_punctuation(old_frag) == self._strip_soft_punctuation(new_frag):
            return False, "punctuation_only_change"

        if not self.llm.passes_postcheck(original, corrected, old_frag, new_frag):
            return False, "postcheck_failed"

        read_sim = self.nlp.reading_similarity(old_frag, new_frag)
        if read_sim < self.no_dict_min_reading_similarity:
            return False, f"low_reading_similarity={read_sim:.3f}"

        max_edit = max(0, int(self.no_dict_sentence_max_edit_chars or 0))
        if max_edit > 0:
            edit_chars = self._count_edit_chars(original, corrected)
            if edit_chars > max_edit:
                return False, f"edit_chars={edit_chars}>{max_edit}"

        return True, "ok"

    def _cleanup_surface_artifacts(self, text: str) -> Tuple[str, List[str]]:
        updated = text or ""
        if not updated:
            return updated, []

        notes: List[str] = []
        jp = r"\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fffー"

        def _drop_embedded_lower_latin(match: re.Match) -> str:
            token = match.group(1)
            notes.append(f"cleanup_latin:{token}")
            return ""

        # Remove obvious lowercase English artifacts embedded in Japanese context.
        updated = re.sub(
            rf"(?<=[{jp}])\s*([a-z]{{4,}})\s*(?=[{jp}])",
            _drop_embedded_lower_latin,
            updated,
        )

        return updated, notes

    @staticmethod
    def _replace_at_span(text: str, start: int, end: int, replacement: str) -> str:
        return text[:start] + replacement + text[end:]

    def _same_homophone_group(self, w1: str, w2: str) -> bool:
        if not w1 or not w2 or w1 == w2:
            return False
        e1 = self.homophones.get(w1)
        e2 = self.homophones.get(w2)
        if e1 and e2:
            r1 = str(e1.get("reading", "") or "")
            r2 = str(e2.get("reading", "") or "")
            if r1 and r2 and r1 == r2:
                return True
        return False

    def _ai_recheck_choice(
        self,
        sentence: str,
        tokens: List[dict],
        idx: int,
        original_word: str,
        ranked: List[Tuple[str, float, dict]],
        extra_forbidden_pairs: Optional[Set[Tuple[str, str]]] = None,
    ) -> str:
        self.llm_recheck_calls += 1

        left_ctx = "".join(t["surface"] for t in tokens[max(0, idx - 4) : idx])
        right_ctx = "".join(t["surface"] for t in tokens[idx + 1 : idx + 5])

        top = ranked[: self.recheck_max_candidates]
        candidates = [x[0] for x in top]

        candidate_lines = []
        for cand, score, detail in top:
            candidate_lines.append(
                f"- {cand}: total={score:.3f}, read_sim={detail.get('reading_similarity', 0.0):.3f}, emb={detail.get('embedding_similarity', 0.0):.3f}"
            )

        system_prompt = self.config.get(
            "llm_recheck_system_prompt",
            "あなたはASR同音異義語修正の判定器です。KEEP または候補語のどちらか1つだけ返してください。説明禁止。",
        )
        prompt = (
            "以下のASR文脈で、元語をKEEPするか候補に置換するかを1つ選んでください。\n"
            "Word Embeddings観点（意味整合）も加味してください。\n\n"
            f"文全体: {sentence}\n"
            f"元語: {original_word}\n"
            f"左文脈: {left_ctx}\n"
            f"右文脈: {right_ctx}\n"
            "候補:\n"
            + "\n".join(candidate_lines)
            + "\n\n出力形式: KEEP または 候補語を1語のみ"
        )

        raw = self.llm.generate(prompt, system_prompt, max_tokens=120, temperature=0.0, top_p=0.2)
        choice = self.llm.parse_choice(raw, candidates)
        if choice != "KEEP":
            if (original_word, choice) in self.forbidden_replacements:
                choice = "KEEP"
            elif extra_forbidden_pairs and (original_word, choice) in extra_forbidden_pairs:
                choice = "KEEP"
        if choice == "KEEP":
            self.llm_recheck_kept += 1
        else:
            self.llm_recheck_changed += 1
        return choice

    def _run_rule_based_and_ai_recheck(
        self, sentence: str, extra_forbidden_pairs: Optional[Set[Tuple[str, str]]] = None
    ) -> Tuple[str, List[str], bool, bool]:
        current = sentence
        changes: List[str] = []
        dict_applied = False
        llm_applied = False
        extra_forbidden_pairs = extra_forbidden_pairs or set()

        tokens = self.nlp.get_morphemes(current)
        if not tokens:
            return current, changes, dict_applied, llm_applied

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            source_word = tok.get("surface", "")
            if not source_word:
                i += 1
                continue

            entry = self.homophones.get(source_word)
            if not entry:
                i += 1
                continue

            candidates = entry.get("candidates", [])
            if not candidates:
                i += 1
                continue

            ranked: List[Tuple[str, float, dict]] = []
            for cand in candidates:
                if (source_word, cand) in extra_forbidden_pairs:
                    continue
                meta = entry.get("candidate_meta", {}).get(cand, {})
                score, detail = self._score_candidate(current, tokens, i, tok, cand, meta)
                if detail.get("rejected"):
                    continue
                ranked.append((cand, score, detail))

            if not ranked:
                i += 1
                continue

            ranked.sort(key=lambda x: x[1], reverse=True)
            best_word, best_score, _ = ranked[0]
            second_score = ranked[1][1] if len(ranked) > 1 else -9999.0
            margin = best_score - second_score

            replaced = False

            # Branch 1: Rule-based replacement
            if best_score >= self.rule_score_threshold and margin >= self.rule_score_margin:
                start, end = tok["start"], tok["end"]
                new_sentence = self._replace_at_span(current, start, end, best_word)
                if new_sentence != current:
                    print(f"  [RULE] {source_word} -> {best_word} (score={best_score:.2f}, margin={margin:.2f})")
                    current = new_sentence
                    changes.append(f"{source_word}→{best_word}")
                    dict_applied = True
                    replaced = True

                    delta = len(best_word) - (end - start)
                    tokens[i]["surface"] = best_word
                    tokens[i]["end"] = start + len(best_word)
                    for j in range(i + 1, len(tokens)):
                        tokens[j]["start"] += delta
                        tokens[j]["end"] += delta

            # Branch 2: AI recheck when uncertain
            if not replaced:
                if best_score >= self.recheck_min_score_to_call:
                    choice = self._ai_recheck_choice(
                        current,
                        tokens,
                        i,
                        source_word,
                        ranked,
                        extra_forbidden_pairs=extra_forbidden_pairs,
                    )
                else:
                    choice = "KEEP"
                if choice and choice not in ("KEEP", source_word):
                    start, end = tok["start"], tok["end"]
                    new_sentence = self._replace_at_span(current, start, end, choice)
                    if new_sentence != current:
                        print(f"  [AI-RECHECK] {source_word} -> {choice}")
                        current = new_sentence
                        changes.append(f"{source_word}→{choice}")
                        llm_applied = True
                        self.no_dict_pair_verified_accept += 1
                        replaced = True

                        delta = len(choice) - (end - start)
                        tokens[i]["surface"] = choice
                        tokens[i]["end"] = start + len(choice)
                        for j in range(i + 1, len(tokens)):
                            tokens[j]["start"] += delta
                            tokens[j]["end"] += delta

            i += 1

        return current, changes, dict_applied, llm_applied

    def _default_no_dict_prompt(self) -> str:
        return (
            "あなたはASR同音異義語修正の専門家です。\n"
            "同音異義語・近音語による誤りだけを修正してください。\n"
            "言い換え・要約・説明は禁止。\n"
            "出力は <改>[修正文] の1行のみ。"
        )

    def _auto_learn_dictionary_entry(self, wrong_word: str, correct_word: str, sentence: str) -> bool:
        if not self.auto_learn_dictionary:
            return False
        if not wrong_word or not correct_word or wrong_word == correct_word:
            return False
        if not self.homophones_file:
            return False

        reading = self.nlp.get_reading(wrong_word) or self.nlp.get_reading(correct_word) or wrong_word
        reading = self.nlp._to_hiragana(reading)

        group = self.homophone_groups.get(reading)
        if not isinstance(group, dict):
            group = {
                "default_candidate": correct_word,
                "candidates": [],
            }
            self.homophone_groups[reading] = group

        candidates = group.get("candidates", [])
        if not isinstance(candidates, list):
            candidates = []
            group["candidates"] = candidates

        # Build lightweight context keywords from nearby tokens
        toks = self.nlp.get_morphemes(sentence)
        kw: List[str] = []
        for t in toks:
            s = t.get("surface", "")
            if s and len(s) >= 2 and s not in (wrong_word, correct_word):
                kw.append(s)
            if len(kw) >= 4:
                break

        def upsert_candidate(word: str, base_weight: float) -> None:
            for cand in candidates:
                if isinstance(cand, dict) and str(cand.get("word", "")) == word:
                    rules = cand.setdefault("context_rules", {})
                    req = rules.setdefault("require_any", [])
                    for k in kw:
                        if k not in req:
                            req.append(k)
                    cand["weight"] = max(float(cand.get("weight", base_weight) or base_weight), base_weight)
                    return

            candidates.append(
                {
                    "word": word,
                    "pos_tags": [],
                    "context_rules": {
                        "require_any": kw,
                        "require_collocation": [sentence[:40]],
                        "exclude": [],
                    },
                    "weight": base_weight,
                }
            )

        upsert_candidate(wrong_word, 1.0)
        upsert_candidate(correct_word, 1.3)
        group["default_candidate"] = correct_word

        self._save_homophone_groups()
        self._refresh_surface_map()
        return True

    def _run_no_dict_branch(self, sentence: str) -> Tuple[str, List[str], bool]:
        self.no_dict_routed += 1
        self.no_dict_llm_calls += 1

        prompt_template = str(self.config.get("no_dict_prompt", self._default_no_dict_prompt()) or "")
        try:
            prompt = prompt_template.format(input_sentence=sentence)
        except Exception:
            prompt = prompt_template.replace("{input_sentence}", sentence)
        if "{input_sentence}" not in prompt_template and sentence not in prompt:
            prompt = f"{prompt.rstrip()}\n\nInput: {sentence}\n"

        no_dict_max_tokens = int(self.config.get("no_dict_max_completion_tokens", 220))
        model_name = str(getattr(self.llm, "model", "") or "").lower()
        # Qwen reasoning-style outputs often need a larger budget to reach final structured answer.
        if "qwen" in model_name:
            no_dict_max_tokens = max(
                no_dict_max_tokens,
                int(self.config.get("no_dict_reasoning_max_completion_tokens", 800)),
            )

        raw = self.llm.generate_completion(
            prompt,
            max_tokens=no_dict_max_tokens,
            temperature=float(self.config.get("no_dict_temperature", 0.0)),
            top_p=float(self.config.get("no_dict_top_p", 0.3)),
        )
        corrected = self.llm.parse_correction(raw, sentence)
        corrected = self._sanitize_generated_sentence(corrected)
        if corrected == "":
            self.no_dict_llm_kept += 1
            return sentence, [], False

        if corrected == sentence:
            self.no_dict_llm_kept += 1
            return sentence, [], False

        old_frag, new_frag = self.llm.extract_span(sentence, corrected)
        if not old_frag or not new_frag:
            old_frag, new_frag = self.llm.extract_span(sentence, corrected)
        if not old_frag:
            old_frag = sentence
        if not new_frag:
            new_frag = corrected

        valid, reason = self._validate_no_dict_candidate(sentence, corrected, old_frag, new_frag)
        if not valid:
            self.no_dict_llm_rejected += 1
            if reason.startswith("edit_chars"):
                self.no_dict_short_rejects += 1
            print(f"  [NO-DICT-REJECT] {reason}")
            return sentence, [], False

        self.no_dict_llm_changed += 1
        print(f"  [NO-DICT] {old_frag} -> {new_frag}")

        if self.auto_learn_dictionary:
            if self._auto_learn_dictionary_entry(old_frag, new_frag, corrected):
                self.no_dict_auto_learned += 1

        changes = [f"{old_frag}→{new_frag}"]
        return corrected, changes, True

    # ------------------------------------------------------------------
    # Public API expected by runner
    # ------------------------------------------------------------------
    def correct_sentence(self, sentence: str) -> Tuple[str, List[str]]:
        self.total_checked += 1
        self.stats["total_sentences"] += 1
        self._pending_learn_pairs.clear()

        current = sentence
        changes: List[str] = []
        dict_applied = False
        llm_applied = False
        reverse_lock_pairs: Set[Tuple[str, str]] = set()

        seed_current, seed_changes = self._apply_seed_replacements(current)
        if seed_changes:
            current = seed_current
            changes.extend(seed_changes)
            self.seed_map_applied += len(seed_changes)
            dict_applied = True
            print(f"  [SEED] {'; '.join(seed_changes)}")
            for ch in seed_changes:
                if "→" in ch:
                    wrong, correct = ch.split("→", 1)
                    wrong = wrong.strip()
                    correct = correct.strip()
                    if wrong and correct and wrong != correct:
                        reverse_lock_pairs.add((correct, wrong))

        current, mem_changes = self._apply_memory_cache(current)
        if mem_changes:
            self.runtime_map_applied += len(mem_changes)
            print(f"  [MEM]  {'; '.join(mem_changes)}")
            changes.extend(mem_changes)
            for ch in mem_changes:
                if "→" in ch:
                    wrong, correct = ch.split("→", 1)
                    wrong = wrong.strip()
                    correct = correct.strip()
                    if wrong and correct and wrong != correct:
                        reverse_lock_pairs.add((correct, wrong))

        # Optional pure-LLM shortcut
        if self.zero_shot_translation_mode:
            current, nd_changes, nd_applied = self._run_no_dict_branch(current)
            changes.extend(nd_changes)
            llm_applied = llm_applied or nd_applied
        else:
            # Branch A/B: dictionary candidate scoring + AI recheck
            if not self.disable_dictionary_stage:
                current, stage_changes, d_ok, l_ok = self._run_rule_based_and_ai_recheck(
                    current, extra_forbidden_pairs=reverse_lock_pairs
                )
                changes.extend(stage_changes)
                dict_applied = dict_applied or d_ok
                llm_applied = llm_applied or l_ok

            # Branch C: no-dict fallback when unresolved
            need_no_dict = False
            if self.enable_no_dict_detector:
                if not changes:
                    # For unchanged sentences, call no-dict only when residual suspicion remains.
                    need_no_dict = self._has_residual_suspicion(current)
                elif self.no_dict_run_on_residual_suspicious and self._has_residual_suspicion(current):
                    need_no_dict = True

            if need_no_dict:
                current, nd_changes, nd_applied = self._run_no_dict_branch(current)
                changes.extend(nd_changes)
                llm_applied = llm_applied or nd_applied

        current, cleanup_notes = self._cleanup_surface_artifacts(current)
        if cleanup_notes:
            print(f"  [CLEANUP] {'; '.join(cleanup_notes)}")
            changes.extend(cleanup_notes)

        for ch in changes:
            if "→" in ch:
                old, new = ch.split("→", 1)
                self._pending_learn_pairs.append((old.strip(), new.strip(), current))

        if dict_applied:
            self.dict_corrected += 1
        if llm_applied:
            self.llm_corrected += 1
        if not changes:
            self.no_change += 1

        return current, changes

    def learn_from_feedback(self, original: str, corrected: str, expected: str) -> bool:
        if not self.config.get("auto_learn_apply_on_verified_hits", True):
            return False
        if not expected or not self._pending_learn_pairs:
            return False

        learned = False
        for wrong, correct, context_sentence in self._pending_learn_pairs:
            if corrected == expected and correct in expected and wrong not in expected:
                self.verified_runtime_map[wrong] = correct
                self.runtime_map_learned += 1
                learned = True
                if self.auto_learn_dictionary:
                    self._auto_learn_dictionary_entry(wrong, correct, context_sentence)

        if learned:
            self._save_runtime_map()
        return learned

    def postprocess_spoken_style(self, text: str) -> Tuple[str, List[str]]:
        # Keep method for runner compatibility, but disable style normalization.
        return text, []

    def print_stats(self):
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
        print(f"No-dict auto-learned: {self.no_dict_auto_learned}")
        print(f"Seed map applied:     {self.seed_map_applied}")
        print(f"Runtime map learned:  {self.runtime_map_learned}")
        print(f"Runtime map applied:  {self.runtime_map_applied}")
        print(f"LLM recheck calls:    {self.llm_recheck_calls}")
        print(f"LLM recheck KEEP:     {self.llm_recheck_kept}")
        print(f"LLM recheck changed:  {self.llm_recheck_changed}")
        print("=" * 60)
