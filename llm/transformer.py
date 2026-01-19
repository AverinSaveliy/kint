import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Union
import math
import logging

logger = logging.getLogger(__name__)

class MultiHeadSelfAttention(nn.Module):
    """
    Многоголовое самовнимание (Multi-Head Self-Attention) с масштабированным скалярным произведением.
    
    Особенности:
    - Масштабированное скалярное произведение
    - Поддержка attention mask
    - KV кэширование для инференса
    - Dropout для регуляризации
    """
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) должно делиться на heads ({heads}) без остатка")

        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # Для статистики
        self.attention_weights = None

    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Tuple]]]:
        """
        Прямой проход многоголового внимания.

        Args:
            x: тензор формы (B, T, C), где
                B — размер батча,
                T — длина последовательности,
                C — размерность признака (dim).
            attention_mask: тензор формы (B, 1, 1, T) или (B, T), 
                где 0 — видимый токен, -inf — замаскированный.
            kv_cache: кортеж (K_cache, V_cache) для инференса
            return_attention: возвращать ли матрицу внимания

        Returns:
            output: тензор той же формы (B, T, C)
            new_kv_cache: обновленный KV кэш (если включен)
        """
        B, T, C = x.shape

        # Вычисляем Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # каждый формы (B, heads, T, head_dim)

        # Используем KV кэш если доступен
        if kv_cache is not None:
            k_cached, v_cached = kv_cache
            k = torch.cat([k_cached, k], dim=-2)
            v = torch.cat([v_cached, v], dim=-2)

        # Скалярное произведение Q и K с масштабированием
        scores = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, T_q, T_k)

        # Применение маски (если задана)
        if attention_mask is not None:
            # Приводим маску к правильной форме
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores + attention_mask

        # Нормализация по последнему измерению (softmax по ключам)
        attn = F.softmax(scores, dim=-1)
        self.attention_weights = attn  # Сохраняем для анализа
        
        attn = self.attn_dropout(attn)

        # Взвешенное суммирование значений V
        out = attn @ v  # (B, heads, T_q, head_dim)
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)

        # Финальная проекция
        out = self.out(out)
        out = self.out_dropout(out)

        if kv_cache is not None:
            new_kv_cache = (k, v)
            if return_attention:
                return out, new_kv_cache, self.attention_weights
            return out, new_kv_cache
        
        if return_attention:
            return out, self.attention_weights
        return out

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Получить последние веса внимания"""
        return self.attention_weights


class TransformerBlock(nn.Module):
    """
    Базовый блок трансформера с Post-LN архитектурой:
        x → Attn → Dropout → Residual →
        x → FFN → Dropout → Residual

    Параметры:
        dim (int): размерность входного вектора.
        heads (int): число голов внимания.
        mlp_ratio (int): коэффициент расширения в MLP (по умолчанию 4).
        dropout (float): вероятность dropout (по умолчанию 0.1).
        use_pre_ln (bool): использовать Pre-LN вместо Post-LN (по умолчанию False).
    """
    def __init__(
        self, 
        dim: int, 
        heads: int, 
        mlp_ratio: int = 4, 
        dropout: float = 0.1,
        use_pre_ln: bool = False,
        activation: str = "gelu"
    ):
        super().__init__()
        self.use_pre_ln = use_pre_ln
        
        # Внимание
        self.attn = MultiHeadSelfAttention(dim, heads, dropout)
        
        # FFN (Feed Forward Network)
        mlp_hidden_dim = dim * mlp_ratio
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Нормализация
        if use_pre_ln:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _get_activation(name: str):
        """Получить функцию активации по имени"""
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swish': nn.SiLU(),
            'glu': nn.GLU(dim=-1)
        }
        if name not in activations:
            raise ValueError(f"Неизвестная активация: {name}")
        return activations[name]

    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple]:
        """
        Прямой проход блока трансформера.

        Args:
            x: входной тензор формы (B, T, dim).
            attention_mask: маска внимания
            kv_cache: кэш для ускорения инференса
            return_attention: возвращать веса внимания

        Returns:
            output: выходной тензор той же формы
            attention_weights: матрица внимания (если return_attention=True)
        """
        # === Блок внимания ===
        if self.use_pre_ln:
            # Pre-LN: нормализация ДО внимания
            attn_input = self.norm1(x)
            attn_output = self.attn(attn_input, attention_mask, kv_cache, return_attention)
            
            if isinstance(attn_output, tuple):
                attn_out, new_kv = attn_output[0], attn_output[1] if len(attn_output) > 1 else None
            else:
                attn_out = attn_output
                new_kv = None
        else:
            # Post-LN: нормализация ПОСЛЕ внимания
            attn_output = self.attn(x, attention_mask, kv_cache, return_attention)
            
            if isinstance(attn_output, tuple):
                attn_out = attn_output[0]
                new_kv = attn_output[1] if len(attn_output) > 1 else None
            else:
                attn_out = attn_output
                new_kv = None
            
            attn_out = self.norm1(attn_out)
        
        # Residual соединение
        x = x + attn_out

        # === FFN блок ===
        if self.use_pre_ln:
            ffn_input = self.norm2(x)
            ffn_out = self.ffn(ffn_input)
        else:
            ffn_out = self.ffn(x)
            ffn_out = self.norm2(ffn_out)

        # Residual соединение
        x = x + ffn_out

        if return_attention and self.attn.attention_weights is not None:
            return x, self.attn.attention_weights
        
        if new_kv is not None:
            return x, new_kv
        
        return x


class TransformerEncoder(nn.Module):
    """
    Полный Transformer энкодер из нескольких блоков.
    
    Параметры:
        dim: размерность скрытого состояния
        num_layers: количество трансформер блоков
        num_heads: количество голов внимания
        mlp_ratio: коэффициент расширения в MLP
        dropout: вероятность dropout
        use_pre_ln: использовать Pre-LN архитектуру
        max_seq_len: максимальная длина последовательности для позиционных эмбеддингов
    """
    def __init__(
        self,
        dim: int,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        use_pre_ln: bool = False,
        max_seq_len: int = 2048,
        add_pos_emb: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.add_pos_emb = add_pos_emb
        
        # Позиционные эмбеддинги
        if add_pos_emb:
            self.pos_embedding = nn.Embedding(max_seq_len, dim)
        
        # Трансформер блоки
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_pre_ln=use_pre_ln
            )
            for _ in range(num_layers)
        ])
        
        # Финальная нормализация
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
        return_all_layers: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Прямой проход энкодера.
        
        Args:
            x: входной тензор (B, T, dim)
            attention_mask: маска внимания
            kv_caches: список KV кэшей для каждого слоя
            return_all_layers: возвращать выходы всех слоев
            return_attention: возвращать веса внимания
            
        Returns:
            Словарь с выходными данными
        """
        B, T, D = x.shape
        
        # Добавить позиционные эмбеддинги
        if self.add_pos_emb:
            pos_ids = torch.arange(T, device=x.device)
            x = x + self.pos_embedding(pos_ids)
        
        # Инициализировать KV кэши если нужны
        if kv_caches is None:
            kv_caches = [None] * self.num_layers
        
        hidden_states = [x]
        attention_weights = []
        new_kv_caches = []
        
        # Пройти через все слои
        for layer_idx, layer in enumerate(self.layers):
            kv_cache = kv_caches[layer_idx] if kv_caches else None
            
            layer_output = layer(
                x,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                return_attention=return_attention
            )
            
            if isinstance(layer_output, tuple):
                x = layer_output[0]
                if len(layer_output) > 1:
                    if return_attention:
                        attention_weights.append(layer_output[1])
                    else:
                        new_kv_caches.append(layer_output[1])
            else:
                x = layer_output
            
            if return_all_layers:
                hidden_states.append(x)
        
        # Финальная нормализация
        x = self.norm(x)
        
        output = {
            'last_hidden_state': x,
            'hidden_states': hidden_states if return_all_layers else None,
            'attention_weights': attention_weights if return_attention else None,
            'kv_caches': new_kv_caches if new_kv_caches else None
        }
        
        return output


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) - более эффективные позиционные эмбеддинги.
    
    Параметры:
        dim: размерность головы внимания
        max_seq_len: максимальная длина последовательности
        base: основание для вычисления углов (по умолчанию 10000)
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Предвычислить углы
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычислить cos и sin матрицы для RoPE.
        
        Args:
            seq_len: длина последовательности
            device: устройство для тензоров
            
        Returns:
            (cos, sin) матрицы формы (1, 1, seq_len, dim)
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Дублируем частоты для четных и нечетных позиций
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        return cos, sin
