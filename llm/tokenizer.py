from pathlib import Path
import sentencepiece as spm
from typing import List, Optional, Dict, Tuple, Union
import logging
import os
from datetime import datetime
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

class RussianBPETokenizer:
    """
    SentencePiece BPE tokenizer для русского языка с расширенным функционалом.
    Локальный, оффлайн, без внешних зависимостей.
    
    Особенности:
    - Кэширование закодированных результатов
    - Поддержка специальных токенов
    - Статистика использования
    - Валидация модели
    """

    def __init__(self, model_path: str = "tokenizer.model"):
        """
        Инициализация токенизатора.
        
        Args:
            model_path: Путь к файлу модели SentencePiece
            
        Raises:
            FileNotFoundError: Если модель не найдена
        """
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor()

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Модель токенизатора не найдена: {self.model_path}. "
                f"Сначала обучите её с помощью train_tokenizer.py"
            )

        self.sp.load(str(self.model_path))
        
        # Кэширование и статистика
        self._encode_cache: Dict[str, List[int]] = {}
        self._decode_cache: Dict[str, str] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.encode_stats: Dict[str, int] = {
            'total_encodes': 0,
            'cached_encodes': 0,
            'total_tokens': 0
        }
        
        # Специальные токены
        self._setup_special_tokens()
        
        logger.info(f"✅ Токенизатор загружен: {self.model_path}")
        logger.info(f"   Размер словаря: {self.vocab_size}")

    def _setup_special_tokens(self):
        """Настроить специальные токены"""
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': self.unk_id,
            '<bos>': self.bos_id,
            '<eos>': self.eos_id,
            '<cls>': self.sp.piece_to_id('<s>') if self.sp.piece_to_id('<s>') >= 0 else 1,
            '<sep>': self.sp.piece_to_id('</s>') if self.sp.piece_to_id('</s>') >= 0 else 2,
            '<mask>': self.sp.piece_to_id('<mask>') if self.sp.piece_to_id('<mask>') >= 0 else 3,
            '<newline>': self.sp.piece_to_id('<newline>') if self.sp.piece_to_id('<newline>') >= 0 else 4,
            '<tab>': self.sp.piece_to_id('<tab>') if self.sp.piece_to_id('<tab>') >= 0 else 5
        }

    @property
    def vocab_size(self) -> int:
        """Размер словаря"""
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        """ID токена паддинга"""
        return self.sp.pad_id()

    @property
    def bos_id(self) -> int:
        """ID токена начала последовательности"""
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        """ID токена конца последовательности"""
        return self.sp.eos_id()

    @property
    def unk_id(self) -> int:
        """ID неизвестного токена"""
        return self.sp.unk_id()

    def encode(
        self, 
        text: str, 
        out_type: str = "int",
        add_bos: bool = False,
        add_eos: bool = False,
        use_cache: bool = True
    ) -> List[int]:
        """
        Кодирует текст в токены с опциональным кэшированием.
        
        Args:
            text: Входной текст
            out_type: 'int' (список ID) или 'str' (список токенов)
            add_bos: Добавить BOS токен в начало
            add_eos: Добавить EOS токен в конец
            use_cache: Использовать кэш для ускорения
            
        Returns:
            Список ID токенов или список строк
            
        Raises:
            ValueError: Если out_type некорректен
        """
        self.encode_stats['total_encodes'] += 1
        
        # Проверка кэша
        cache_key = f"{text}|{add_bos}|{add_eos}"
        if use_cache and cache_key in self._encode_cache:
            self.encode_stats['cached_encodes'] += 1
            self.cache_hits += 1
            return self._encode_cache[cache_key]
        
        self.cache_misses += 1
        
        if out_type == "int":
            tokens = self.sp.encode(text)
        elif out_type == "str":
            tokens = self.sp.encode(text, out_type=str)
        else:
            raise ValueError("out_type должен быть 'int' или 'str'")
        
        # Добавить специальные токены
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]
        
        # Кэшировать результат
        if use_cache and len(cache_key) < 10000:  # Кэшировать только короткие текста
            self._encode_cache[cache_key] = tokens
        
        self.encode_stats['total_tokens'] += len(tokens)
        return tokens

    def decode(self, tokens: List[int], skip_special: bool = False) -> str:
        """
        Декодирует токены в текст.
        
        Args:
            tokens: Список ID токенов
            skip_special: Пропустить специальные токены при декодировании
            
        Returns:
            Декодированный текст
        """
        if skip_special:
            # Фильтровать специальные токены
            tokens = [
                t for t in tokens 
                if t not in (self.pad_id, self.bos_id, self.eos_id, self.unk_id)
            ]
        
        # Кэширование для часто используемых последовательностей
        cache_key = ",".join(map(str, tokens))
        if cache_key in self._decode_cache:
            self.cache_hits += 1
            return self._decode_cache[cache_key]
        
        self.cache_misses += 1
        result = self.sp.decode(tokens)
        
        if len(cache_key) < 1000:
            self._decode_cache[cache_key] = result
        
        return result

    def tokenize(self, text: str, as_ids: bool = False) -> Union[List[str], List[int]]:
        """
        Возвращает список токенов (строки или ID).
        
        Args:
            text: Входной текст
            as_ids: Если True, вернуть ID вместо строк
            
        Returns:
            Список токенов
        """
        if as_ids:
            return self.encode(text, out_type="int")
        else:
            return self.encode(text, out_type="str")

    def detokenize(self, tokens: List[str]) -> str:
        """
        Собирает текст из списка токенов-строк.
        
        Args:
            tokens: Список токенов-строк
            
        Returns:
            Декодированный текст
        """
        token_ids = [self.sp.piece_to_id(t) for t in tokens]
        return self.decode(token_ids)

    def get_token_info(self, token_id: int) -> Dict[str, Union[str, int, float]]:
        """
        Получить информацию о токене.
        
        Args:
            token_id: ID токена
            
        Returns:
            Информация о токене
        """
        piece = self.sp.id_to_piece(token_id)
        return {
            'id': token_id,
            'piece': piece,
            'is_control': self.sp.is_control(token_id),
            'is_unknown': self.sp.is_unknown(token_id),
            'is_unused': self.sp.is_unused(token_id),
            'byte_length': len(piece.encode('utf-8'))
        }

    def get_vocab_stats(self) -> Dict[str, Union[int, float]]:
        """
        Получить статистику словаря.
        
        Returns:
            Словарь со статистикой
        """
        control_tokens = sum(1 for i in range(self.vocab_size) if self.sp.is_control(i))
        unused_tokens = sum(1 for i in range(self.vocab_size) if self.sp.is_unused(i))
        
        return {
            'vocab_size': self.vocab_size,
            'control_tokens': control_tokens,
            'unused_tokens': unused_tokens,
            'regular_tokens': self.vocab_size - control_tokens - unused_tokens,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) 
                            if (self.cache_hits + self.cache_misses) > 0 else 0,
            'total_encodes': self.encode_stats['total_encodes'],
            'total_tokens_encoded': self.encode_stats['total_tokens'],
            'avg_tokens_per_encode': self.encode_stats['total_tokens'] / max(1, self.encode_stats['total_encodes'])
        }

    def validate_text(self, text: str) -> Tuple[bool, List[str]]:
        """
        Валидировать текст для кодирования.
        
        Args:
            text: Входной текст
            
        Returns:
            (is_valid, issues) - кортеж валидности и списка проблем
        """
        issues = []
        
        if not text:
            issues.append("Пустой текст")
            return False, issues
        
        if len(text) > 1_000_000:
            issues.append(f"Текст слишком длинный: {len(text)} символов")
        
        # Проверка на контрольные символы
        if any(ord(c) < 32 and c not in '\n\t\r' for c in text):
            issues.append("Обнаружены контрольные символы")
        
        # Проверка на невалидные UTF-8
        try:
            text.encode('utf-8').decode('utf-8')
        except UnicodeDecodeError:
            issues.append("Невалидное UTF-8 кодирование")
            return False, issues
        
        return len(issues) == 0, issues

    def get_encoding_info(self, text: str) -> Dict[str, Union[int, float, List]]:
        """
        Получить детальную информацию о кодировании текста.
        
        Args:
            text: Входной текст
            
        Returns:
            Информация о кодировании
        """
        tokens = self.encode(text, out_type="int", use_cache=False)
        token_pieces = self.encode(text, out_type="str", use_cache=False)
        
        # Статистика по длинам токенов
        token_lengths = [len(piece.encode('utf-8')) for piece in token_pieces]
        
        return {
            'text_length_chars': len(text),
            'text_length_bytes': len(text.encode('utf-8')),
            'num_tokens': len(tokens),
            'compression_ratio': len(text) / len(tokens) if tokens else 0,
            'avg_token_length': sum(token_lengths) / len(token_lengths) if token_lengths else 0,
            'min_token_length': min(token_lengths) if token_lengths else 0,
            'max_token_length': max(token_lengths) if token_lengths else 0,
            'token_ids': tokens[:50],  # Первые 50 токенов
            'token_pieces': token_pieces[:50]
        }

    def clear_cache(self):
        """Очистить кэш"""
        self._encode_cache.clear()
        self._decode_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("✅ Кэш токенизатора очищен")

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Получить статистику кэша"""
        total = self.cache_hits + self.cache_misses
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_lookups': total,
            'hit_rate': self.cache_hits / total if total > 0 else 0,
            'encode_cache_size': len(self._encode_cache),
            'decode_cache_size': len(self._decode_cache)
        }

    def save_stats(self, output_path: str = "tokenizer_stats.json"):
        """
        Сохранить статистику в JSON файл.
        
        Args:
            output_path: Путь для сохранения
        """
        stats = {
            'timestamp': datetime.now().isoformat(),
            'vocab_stats': self.get_vocab_stats(),
            'cache_stats': self.get_cache_stats(),
            'encode_stats': self.encode_stats
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Статистика сохранена: {output_path}")

    def benchmark_encode_decode(self, text: str, iterations: int = 100) -> Dict[str, float]:
        """
        Бенчмарк для кодирования/декодирования.
        
        Args:
            text: Тестовый текст
            iterations: Количество итераций
            
        Returns:
            Результаты бенчмарка
        """
        import time
        
        # Бенчмарк кодирования
        start = time.time()
        for _ in range(iterations):
            tokens = self.encode(text, use_cache=False)
        encode_time = (time.time() - start) / iterations
        
        # Бенчмарк декодирования
        tokens = self.encode(text)
        start = time.time()
        for _ in range(iterations):
            _ = self.decode(tokens)
        decode_time = (time.time() - start) / iterations
        
        return {
            'encode_time_ms': encode_time * 1000,
            'decode_time_ms': decode_time * 1000,
            'encode_throughput_chars_per_sec': len(text) / encode_time,
            'decode_throughput_tokens_per_sec': len(tokens) / decode_time
        }
