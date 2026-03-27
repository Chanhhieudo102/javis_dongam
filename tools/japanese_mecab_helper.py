#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Japanese MeCab Integration Helper
日本語形態素解析ヘルパー

Provides morphological analysis for Japanese text using MeCab.
MeCabを使用した日本語のテキストの形態素解析を提供します。

Requirements:
    pip install mecab-python3 unidic-lite
    # Or for full unidic:
    # pip install mecab-python3 unidic
    # python -m unidic download
"""

import sys
from typing import List, Dict, Tuple, Optional

try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    print("Warning: MeCab not installed. Install with: pip install mecab-python3 unidic-lite")


class JapaneseMorphAnalyzer:
    """Japanese Morphological Analyzer using MeCab"""
    
    def __init__(self, dicdir: Optional[str] = None):
        """
        Initialize MeCab analyzer
        
        Args:
            dicdir: Dictionary directory path (optional)
        """
        if not MECAB_AVAILABLE:
            raise ImportError("MeCab is not installed. Install with: pip install mecab-python3 unidic-lite")
        
        try:
            if dicdir:
                self.tagger = MeCab.Tagger(f'-d {dicdir}')
            else:
                # Try to use unidic-lite by default
                self.tagger = MeCab.Tagger()
            self.available = True
        except Exception as e:
            print(f"Warning: Failed to initialize MeCab: {e}")
            self.available = False
    
    def tokenize(self, text: str) -> List[Dict[str, str]]:
        """
        Tokenize Japanese text into morphemes
        
        Args:
            text: Input Japanese text
            
        Returns:
            List of dictionaries containing token information
            
        Example:
            >>> analyzer = JapaneseMorphAnalyzer()
            >>> tokens = analyzer.tokenize("東京へ行きます")
            >>> for token in tokens:
            ...     print(token['surface'], token['pos'], token['base'])
        """
        if not self.available:
            return []
        
        result = []
        node = self.tagger.parseToNode(text)
        
        while node:
            if node.surface:  # Skip BOS/EOS
                features = node.feature.split(',')
                token_info = {
                    'surface': node.surface,          # 表層形
                    'pos': features[0] if len(features) > 0 else '',      # 品詞
                    'pos_detail1': features[1] if len(features) > 1 else '',  # 品詞細分類1
                    'pos_detail2': features[2] if len(features) > 2 else '',  # 品詞細分類2
                    'pos_detail3': features[3] if len(features) > 3 else '',  # 品詞細分類3
                    'conjugation': features[4] if len(features) > 4 else '',  # 活用型
                    'conjugation_form': features[5] if len(features) > 5 else '',  # 活用形
                    'base': features[6] if len(features) > 6 else node.surface,  # 基本形
                    'reading': features[7] if len(features) > 7 else '',  # 読み
                    'pronunciation': features[8] if len(features) > 8 else ''  # 発音
                }
                result.append(token_info)
            node = node.next
        
        return result
    
    def extract_words(self, text: str, pos_filter: Optional[List[str]] = None) -> List[str]:
        """
        Extract words from text, optionally filtering by POS tags
        
        Args:
            text: Input text
            pos_filter: List of POS tags to include (e.g., ['名詞', '動詞'])
            
        Returns:
            List of extracted words
        """
        tokens = self.tokenize(text)
        
        if pos_filter:
            return [token['surface'] for token in tokens if token['pos'] in pos_filter]
        else:
            return [token['surface'] for token in tokens]
    
    def extract_nouns(self, text: str) -> List[str]:
        """Extract all nouns from text"""
        return self.extract_words(text, pos_filter=['名詞'])
    
    def extract_proper_nouns(self, text: str) -> List[str]:
        """Extract proper nouns (固有名詞) from text"""
        tokens = self.tokenize(text)
        return [token['surface'] for token in tokens 
                if token['pos'] == '名詞' and token['pos_detail1'] == '固有名詞']
    
    def identify_person_names(self, text: str) -> List[str]:
        """
        Identify person names in text
        
        Returns:
            List of detected person names
        """
        tokens = self.tokenize(text)
        person_names = []
        
        for token in tokens:
            if (token['pos'] == '名詞' and 
                token['pos_detail1'] == '固有名詞' and 
                token['pos_detail2'] == '人名'):
                person_names.append(token['surface'])
        
        return person_names
    
    def identify_place_names(self, text: str) -> List[str]:
        """
        Identify place names in text
        
        Returns:
            List of detected place names
        """
        tokens = self.tokenize(text)
        place_names = []
        
        for token in tokens:
            if (token['pos'] == '名詞' and 
                token['pos_detail1'] == '固有名詞' and 
                token['pos_detail2'] == '地域'):
                place_names.append(token['surface'])
        
        return place_names
    
    def get_base_forms(self, text: str) -> List[str]:
        """
        Get base forms (lemmas) of all words in text
        
        Args:
            text: Input text
            
        Returns:
            List of base forms
        """
        tokens = self.tokenize(text)
        return [token['base'] for token in tokens if token['base']]
    
    def get_readings(self, text: str) -> List[Tuple[str, str]]:
        """
        Get surface form and reading pairs
        
        Returns:
            List of (surface, reading) tuples
        """
        tokens = self.tokenize(text)
        return [(token['surface'], token['reading']) for token in tokens if token['reading']]


def analyze_text_with_mecab(text: str, verbose: bool = False) -> Dict:
    """
    Analyze Japanese text using MeCab
    
    Args:
        text: Input text
        verbose: Print detailed analysis
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = JapaneseMorphAnalyzer()
    
    tokens = analyzer.tokenize(text)
    nouns = analyzer.extract_nouns(text)
    proper_nouns = analyzer.extract_proper_nouns(text)
    person_names = analyzer.identify_person_names(text)
    place_names = analyzer.identify_place_names(text)
    
    result = {
        'original_text': text,
        'tokens': tokens,
        'token_count': len(tokens),
        'nouns': nouns,
        'proper_nouns': proper_nouns,
        'person_names': person_names,
        'place_names': place_names
    }
    
    if verbose:
        print(f"\n=== Analysis of: {text} ===")
        print(f"Tokens: {len(tokens)}")
        print("\nTokenization:")
        for token in tokens:
            print(f"  {token['surface']:<10} {token['pos']:<6} {token['base']:<10} {token['reading']}")
        print(f"\nNouns: {', '.join(nouns)}")
        print(f"Proper Nouns: {', '.join(proper_nouns)}")
        print(f"Person Names: {', '.join(person_names)}")
        print(f"Place Names: {', '.join(place_names)}")
    
    return result


# Fallback simple tokenizer when MeCab is not available
def simple_japanese_tokenize(text: str) -> List[str]:
    """
    Simple character-based tokenization (fallback when MeCab is unavailable)
    
    Args:
        text: Input text
        
    Returns:
        List of characters/tokens
    """
    import re
    
    # Split by spaces and punctuation
    tokens = []
    current = ""
    
    for char in text:
        if char in " \t\n、。！？「」（）【】":
            if current:
                tokens.append(current)
                current = ""
            if char.strip():
                tokens.append(char)
        else:
            current += char
    
    if current:
        tokens.append(current)
    
    return tokens


if __name__ == "__main__":
    # Example usage
    print("Japanese MeCab Integration Helper")
    print("=" * 50)
    
    if not MECAB_AVAILABLE:
        print("\nMeCab is not installed!")
        print("Install with: pip install mecab-python3 unidic-lite")
        print("\nUsing simple fallback tokenizer:")
        
        test_text = "佐藤さんは大阪へ行きました"
        tokens = simple_japanese_tokenize(test_text)
        print(f"Text: {test_text}")
        print(f"Tokens: {tokens}")
        sys.exit(1)
    
    # Test examples
    test_texts = [
        "佐藤さんは大阪へ行きました",
        "田中さんが感光地を訪れた",
        "大坂城は有名な観光地です",
        "彼は会議で意見を述べた",
        "橋を使ってご飯を食べます"
    ]
    
    analyzer = JapaneseMorphAnalyzer()
    
    for text in test_texts:
        print(f"\n{'=' * 60}")
        analyze_text_with_mecab(text, verbose=True)
    
    print(f"\n{'=' * 60}")
    print("MeCab integration test completed!")
