"""
KINT - Мегаинтеллект с квантовыми компонентами
"""
__version__ = "1.0.0"
__author__ = "KINT Development"

from llm.model import KINTLanguageModel
from llm.tokenizer import RussianBPETokenizer
from llm.transformer import TransformerBlock, MultiHeadSelfAttention

__all__ = [
    "KINTLanguageModel",
    "RussianBPETokenizer",
    "TransformerBlock",
    "MultiHeadSelfAttention"
]
