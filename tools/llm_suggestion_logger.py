#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Suggestion Logger & Verifier
Logs LLM suggestions with confidence scores, verification status, and metadata
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class LLMSuggestionLogger:
    """Log and verify LLM-generated suggestions for error corrections"""
    
    def __init__(self, log_dir="result/llm_suggestions"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"suggestions_{timestamp}.jsonl")
        self.stats_file = os.path.join(log_dir, f"stats_{timestamp}.json")
        
        self.stats = {
            "total_suggestions": 0,
            "dictionary_matches": 0,
            "llm_suggestions": 0,
            "verified_correct": 0,
            "verified_incorrect": 0,
            "unverified": 0,
            "kept_original": 0,
        }
    
    def log_suggestion(self, 
                      pipeline_type: str,
                      sentence: str,
                      error_location: str,
                      error_text: str,
                      suggestion_type: str,  # "DICT" | "LLM_CHOICE" | "LLM_NEW"
                      suggestion: str,
                      confidence: float,
                      tokens_used: Optional[Dict] = None,
                      verification_status: str = "unverified",
                      fallback_reason: Optional[str] = None) -> None:
        """
        Log a suggestion with full metadata
        
        Args:
            pipeline_type: 'kanji' | 'homophones' | 'proper_nouns'
            sentence: Original sentence
            error_location: Position or description of error
            error_text: The erroneous text
            suggestion_type: Type of suggestion (DICT, LLM_CHOICE, LLM_NEW)
            suggestion: The suggested correction
            confidence: Confidence score (0-10)
            tokens_used: Dict with prompt_tokens, completion_tokens, total_tokens
            verification_status: 'verified_correct' | 'verified_incorrect' | 'unverified'
            fallback_reason: Reason if fallback to original was used
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "pipeline": pipeline_type,
            "sentence": sentence,
            "error_location": error_location,
            "error_text": error_text,
            "suggestion_type": suggestion_type,
            "suggestion": suggestion,
            "confidence": confidence,
            "tokens": tokens_used or {},
            "verification_status": verification_status,
            "fallback_reason": fallback_reason,
        }
        
        # Append to JSONL log
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # Update stats
        self.stats["total_suggestions"] += 1
        if suggestion_type == "DICT":
            self.stats["dictionary_matches"] += 1
        elif suggestion_type.startswith("LLM"):
            self.stats["llm_suggestions"] += 1
        
        if suggestion == error_text:
            self.stats["kept_original"] += 1
        elif verification_status == "verified_correct":
            self.stats["verified_correct"] += 1
        elif verification_status == "verified_incorrect":
            self.stats["verified_incorrect"] += 1
        else:
            self.stats["unverified"] += 1
    
    def verify_suggestion(self, 
                         sentence: str, 
                         suggestion: str,
                         reference: Optional[str] = None) -> Tuple[str, float]:
        """
        Basic rule-based verification of suggestion
        
        Returns: (status, score)
        - status: 'verified_correct' | 'verified_incorrect' | 'unverified'
        - score: 0-10 confidence
        """
        # Rule 1: Check if suggestion is in sentence (context match)
        if suggestion in sentence:
            return "verified_correct", 8.0
        
        # Rule 2: Check if suggestion is a known common correction
        if reference and suggestion == reference:
            return "verified_correct", 7.0
        
        # Rule 3: Check if suggestion doesn't introduce new errors
        # (e.g., length change within reasonable bounds)
        if len(suggestion) > len(sentence) * 1.5:
            return "verified_incorrect", 2.0
        
        # Rule 4: Check for nonsensical/hallucinated characters
        if "\n" in suggestion or len(suggestion) > 20:
            return "verified_incorrect", 1.0
        
        # Default: unverified (needs human review)
        return "unverified", 5.0
    
    def save_stats(self) -> None:
        """Save accumulated stats to JSON file"""
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
    
    def load_and_review(self, limit: int = 50) -> List[Dict]:
        """Load recent suggestions for review"""
        suggestions = []
        if not os.path.exists(self.log_file):
            return []
        
        with open(self.log_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                suggestions.append(json.loads(line))
        
        return suggestions
    
    def print_summary(self) -> None:
        """Print summary statistics"""
        print("\n" + "=" * 70)
        print("LLM SUGGESTION SUMMARY")
        print("=" * 70)
        print(f"Total suggestions:       {self.stats['total_suggestions']}")
        print(f"  - Dictionary matches:  {self.stats['dictionary_matches']}")
        print(f"  - LLM suggestions:     {self.stats['llm_suggestions']}")
        print(f"Outcomes:")
        print(f"  - Verified correct:    {self.stats['verified_correct']}")
        print(f"  - Verified incorrect:  {self.stats['verified_incorrect']}")
        print(f"  - Kept original:       {self.stats['kept_original']}")
        print(f"  - Unverified:          {self.stats['unverified']}")
        print("=" * 70)


class LLMSuggestionFormatter:
    """Utility to format prompts with few-shot examples for LLM suggestions"""
    
    @staticmethod
    def few_shot_kanji() -> str:
        """Few-shot examples for kanji correction"""
        return """【修正例】
入力: 自分の意見に固丸する
選択肢: 固丸, 固執, KEEP_ORIGINAL
出力形式: CHOICE: 固執

入力: 計画を逐行した
選択肢: 逐行, 遂行, KEEP_ORIGINAL
出力形式: CHOICE: 遂行

入力: 音声認識で「学建」と誤認識（実は「学研」で正しい）
選択肢: 学建, 学研, KEEP_ORIGINAL
出力形式: CHOICE: KEEP_ORIGINAL"""
    
    @staticmethod
    def few_shot_homophones() -> str:
        """Few-shot examples for homophone correction"""
        return """【修正例】
入力: 私は学生の時代に戻りたい
対象語: 時代 (じだい)
選択肢: 時代, 時大, KEEP_ORIGINAL
出力形式: CHOICE: 時代

入力: これは重大な問題だ
対象語: 重大
選択肢: 重大, 住宅, KEEP_ORIGINAL
出力形式: CHOICE: 重大

入力: 政策が必要だ
対象語: 政策
選択肢: 政策, 制作, KEEP_ORIGINAL
出力形式: CHOICE: 政策"""
    
    @staticmethod
    def few_shot_with_new_suggestions() -> str:
        """Few-shot examples when allow_llm_suggest=True"""
        return """【新規提案を認める場合の修正例】
入力: 耳元で大きな音が鳴った
辞書の候補: [音, おんせい]  # 辞書にない可能性
出力形式1(辞書内): CHOICE: 音
出力形式2(新規提案): NEW: 音量  # 「音量」がより自然な場合

入力: 彼は正確に作業を進めた
辞書の候補: [正確, 正式]
出力形式1(辞書内): CHOICE: 正確
出力形式2(不確実): CHOICE: KEEP_ORIGINAL  # 確信がない場合"""
    
    @staticmethod
    def format_prompt_with_few_shot(base_prompt: str, 
                                   few_shot_examples: str,
                                   allow_llm_suggest: bool = False) -> str:
        """
        Insert few-shot examples into base prompt
        
        Args:
            base_prompt: Original prompt template
            few_shot_examples: Few-shot examples string
            allow_llm_suggest: Whether to include NEW suggestion format
        
        Returns: Enhanced prompt with examples
        """
        if allow_llm_suggest:
            return f"""{base_prompt}

{LLMSuggestionFormatter.few_shot_with_new_suggestions()}

【出力ルール】
- 辞書内の候補から選ぶ場合: CHOICE: <候補>
- 新しい提案をする場合: NEW: <提案>  (確信がある場合のみ)
- 元の語を保つ: CHOICE: KEEP_ORIGINAL
- 説明は絶対禁止
"""
        else:
            return f"""{base_prompt}

{few_shot_examples}

【出力ルール】
- 辞書内の候補から選ぶ: CHOICE: <候補>
- 元の語を保つ: CHOICE: KEEP_ORIGINAL
- 説明は絶対禁止
"""


if __name__ == "__main__":
    # Test
    logger = LLMSuggestionLogger()
    
    # Log some test suggestions
    logger.log_suggestion(
        pipeline_type="kanji",
        sentence="自分の意見に固丸する",
        error_location="pos 4",
        error_text="固丸",
        suggestion_type="DICT",
        suggestion="固執",
        confidence=9.0,
        verification_status="verified_correct"
    )
    
    logger.log_suggestion(
        pipeline_type="homophones",
        sentence="これは重大な問題だ",
        error_location="word 2",
        error_text="重大",
        suggestion_type="LLM_CHOICE",
        suggestion="重大",
        confidence=7.5,
        verification_status="unverified"
    )
    
    logger.save_stats()
    logger.print_summary()
    
    print("Sample suggestions loaded:")
    suggestions = logger.load_and_review(limit=5)
    for s in suggestions:
        print(f"  {s['pipeline']}: {s['error_text']} → {s['suggestion']} (conf: {s['confidence']})")
