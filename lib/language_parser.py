"""
Language Parser for SOCT-SAM (Stochastic-Order Constrained Transport).

Extracts semantic roles from referring expressions:
1. Target phrase (T): Target object noun phrase and its modifying attributes.
2. Contextual Anchors (A_m): Reference object noun phrase.
3. Directional Relations (d_m): Supported spatial directions:
   - 'left'   (target is to the left of anchor)
   - 'right'  (target is to the right of anchor)
   - 'above' / 'top' / 'north' (target is above anchor)
   - 'below' / 'bottom' / 'south' (target is below anchor)

Fallback:
Deterministic whole-content target with no ordering constraints if no
directional relation is identified or if parsing fails.
"""

import re
from typing import List, Dict, Tuple, Optional

# Supported canonical directions
DIRECTION_MAP = {
    'left': 'left',
    'to the left': 'left',
    'to the left of': 'left',
    'on the left': 'left',
    'on the left of': 'left',
    'west': 'left',
    'to the west': 'left',
    'to the west of': 'left',

    'right': 'right',
    'to the right': 'right',
    'to the right of': 'right',
    'on the right': 'right',
    'on the right of': 'right',
    'east': 'right',
    'to the east': 'right',
    'to the east of': 'right',

    'above': 'above',
    'over': 'above',
    'top': 'above',
    'on top of': 'above',
    'north': 'above',
    'to the north': 'above',
    'to the north of': 'above',
    'upper': 'above',

    'below': 'below',
    'under': 'below',
    'underneath': 'below',
    'beneath': 'below',
    'bottom': 'below',
    'south': 'below',
    'to the south': 'below',
    'to the south of': 'below',
    'lower': 'below',
}

# Stopwords / Function words to exclude from visual mass matching columns
FUNCTION_WORDS = {
    'a', 'an', 'the', 'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'of', 'and', 'or', 'but'
}

# Pre-compiled regex patterns for directional splits
RELATION_PATTERNS = [
    # "X to the left of Y", "X on the left of Y", etc.
    (re.compile(r'^(.*?)\s+(?:to\s+the\s+left\s+of|on\s+the\s+left\s+of|to\s+the\s+left|on\s+the\s+left)\s+(.*?)$', re.IGNORECASE), 'left'),
    (re.compile(r'^(.*?)\s+(?:to\s+the\s+right\s+of|on\s+the\s+right\s+of|to\s+the\s+right|on\s+the\s+right)\s+(.*?)$', re.IGNORECASE), 'right'),
    (re.compile(r'^(.*?)\s+(?:to\s+the\s+north\s+of|north\s+of)\s+(.*?)$', re.IGNORECASE), 'above'),
    (re.compile(r'^(.*?)\s+(?:to\s+the\s+south\s+of|south\s+of)\s+(.*?)$', re.IGNORECASE), 'below'),
    (re.compile(r'^(.*?)\s+(?:to\s+the\s+west\s+of|west\s+of)\s+(.*?)$', re.IGNORECASE), 'left'),
    (re.compile(r'^(.*?)\s+(?:to\s+the\s+east\s+of|east\s+of)\s+(.*?)$', re.IGNORECASE), 'right'),
    (re.compile(r'^(.*?)\s+(?:above|over|on\s+top\s+of)\s+(.*?)$', re.IGNORECASE), 'above'),
    (re.compile(r'^(.*?)\s+(?:below|under|underneath|beneath)\s+(.*?)$', re.IGNORECASE), 'below'),
    (re.compile(r'^(.*?)\s+left\s+(?:of|to)\s+(.*?)$', re.IGNORECASE), 'left'),
    (re.compile(r'^(.*?)\s+right\s+(?:of|to)\s+(.*?)$', re.IGNORECASE), 'right'),
]


class ReferringExpressionParser:
    """
    Parser to identify target phrases, contextual anchor phrases,
    and directional relations from referring expressions.
    Includes a deterministic cache to avoid repeated parsing overhead.
    """
    def __init__(self):
        self._cache: Dict[str, Dict] = {}

    def parse(self, text: str) -> Dict:
        """
        Parses text expression into:
        - target_text: str
        - relations: List[Tuple[anchor_text, direction]]
        - is_fallback: bool
        """
        cleaned = text.strip()
        if cleaned in self._cache:
            return self._cache[cleaned]

        res = self._parse_expression(cleaned)
        self._cache[cleaned] = res
        return res

    def _parse_expression(self, text: str) -> Dict:
        # Check against relational patterns
        for pattern, direction in RELATION_PATTERNS:
            match = pattern.match(text)
            if match:
                target_part = match.group(1).strip()
                anchor_part = match.group(2).strip()
                if target_part and anchor_part:
                    return {
                        'target_text': target_part,
                        'relations': [(anchor_part, direction)],
                        'is_fallback': False,
                        'raw': text,
                    }

        # Fallback: Whole expression is treated as target, no relations
        return {
            'target_text': text,
            'relations': [],
            'is_fallback': True,
            'raw': text,
        }

    def align_tokens_to_roles(
        self,
        full_tokens: List[str],
        parsed_info: Dict,
    ) -> Tuple[List[int], List[Tuple[List[int], str]]]:
        """
        Aligns token list (subwords / words) to target indices and anchor indices.

        Args:
            full_tokens: List of tokens as strings (length = seq_len).
            parsed_info: Output of self.parse(text).

        Returns:
            target_indices: List[int] indices in full_tokens corresponding to Target.
            anchor_relations: List[Tuple[List[int], str]], each is (anchor_indices, direction).
        """
        seq_len = len(full_tokens)
        if parsed_info['is_fallback'] or not parsed_info['relations']:
            # All non-punctuation/non-special tokens are target tokens
            target_indices = []
            for idx, tok in enumerate(full_tokens):
                cleaned = tok.lower().strip('#').strip()
                if cleaned and cleaned not in FUNCTION_WORDS and cleaned not in {'[cls]', '[sep]', '<s>', '</s>', '<pad>'}:
                    target_indices.append(idx)
            if not target_indices:
                target_indices = list(range(seq_len))
            return target_indices, []

        target_text = parsed_info['target_text'].lower()
        target_words = set(re.findall(r'\w+', target_text))

        target_indices = []
        anchor_relations = []

        for idx, tok in enumerate(full_tokens):
            cleaned = tok.lower().strip('#').strip()
            if not cleaned or cleaned in {'[cls]', '[sep]', '<s>', '</s>', '<pad>'}:
                continue
            if cleaned in target_words and cleaned not in FUNCTION_WORDS:
                target_indices.append(idx)

        # Process each relation
        for anchor_text, direction in parsed_info['relations']:
            anchor_words = set(re.findall(r'\w+', anchor_text.lower()))
            anchor_indices = []
            for idx, tok in enumerate(full_tokens):
                cleaned = tok.lower().strip('#').strip()
                if not cleaned or cleaned in {'[cls]', '[sep]', '<s>', '</s>', '<pad>'}:
                    continue
                if cleaned in anchor_words and cleaned not in FUNCTION_WORDS:
                    anchor_indices.append(idx)
            if anchor_indices:
                anchor_relations.append((anchor_indices, direction))

        # Safeguard: if target_indices empty, use all tokens before relation keyword
        if not target_indices:
            for idx, tok in enumerate(full_tokens):
                cleaned = tok.lower().strip('#').strip()
                if cleaned and cleaned not in FUNCTION_WORDS and cleaned not in {'[cls]', '[sep]', '<s>', '</s>', '<pad>'}:
                    target_indices.append(idx)
            if not target_indices:
                target_indices = list(range(seq_len))

        return target_indices, anchor_relations


# Global shared parser instance
GLOBAL_PARSER = ReferringExpressionParser()
