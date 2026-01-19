import sentencepiece as spm
from pathlib import Path
import re
import json
from typing import Optional, List, Dict, Tuple
import logging
from datetime import datetime
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Пути по умолчанию
DATA_PATH = Path("data/corpus.txt")
MODEL_PREFIX = "tokenizer"
STATS_FILE = "tokenizer_stats.json"

class TextCleaner:
    """Очистка и нормализация текста с расширенным функционалом"""
    
    def __init__(self, aggressive: bool = False):
        """
        Инициализация очистителя текста.
        
        Args:
            aggressive: Если True, применяет более агрессивную очистку
        """
        self.aggressive = aggressive
        self.stats = {
            'lines_processed': 0,
            'lines_cleaned': 0,
            'chars_removed': 0,
            'avg_line_length_before': 0,
            'avg_line_length_after': 0
        }
    
    def clean_text(self, line: str) -> str:
        """
        Очищает строку от шума и нормализует.
        
        Args:
            line: Входная строка
            
        Returns:
            Очищенная строка
        """
        original_len = len(line)
        
        # Удаляем лишние пробелы
        line = re.sub(r'\s+', ' ', line)
        
        # Удаляем специальные символы (опционально)
        if self.aggressive:
            # Удаляем одиночные цифры
            line = re.sub(r'\b\d\b', '', line)
            # Удаляем короткие английские подстроки
            line = re.sub(r'\b[a-zA-Z]{1,2}\b', '', line)
        
        # Нормализуем кавычки
        line = line.replace('«', '"').replace('»', '"')
        line = line.replace(''', "'").replace(''', "'")
        line = line.replace('„', '"').replace('"', '"')
        
        # Нормализуем тире
        line = line.replace('–', '-').replace('—', '-')
        
        # Удаляем контрольные символы (кроме пробельных)
        line = ''.join(c for c in line if ord(c) >= 32 or c in '\n\t\r')
        
        # Удаляем лишние пробелы в начале и конце
        line = line.strip()
        
        self.stats['chars_removed'] += original_len - len(line)
        
        return line
    
    def is_valid_line(self, line: str, min_length: int = 5) -> bool:
        """
        Проверить, валидна ли строка для обучения.
        
        Args:
            line: Строка для проверки
            min_length: Минимальная длина строки
            
        Returns:
            True если валидна
        """
        if not line:
            return False
        
        if len(line) < min_length:
            return False
        
        # Не менее 50% букв в строке
        letter_count = sum(1 for c in line if c.isalpha())
        if letter_count / len(line) < 0.5:
            return False
        
        # Не более 90% одного символа
        char_counts = {}
        for c in line:
            char_counts[c] = char_counts.get(c, 0) + 1
        max_char_ratio = max(char_counts.values()) / len(line)
        if max_char_ratio > 0.9:
            return False
        
        return True
    
    def get_stats(self) -> Dict:
        """Получить статистику очистки"""
        return self.stats.copy()

def estimate_vocab_size(text_size: int) -> int:
    """
    Эвристика для оптимального размера словаря на основе размера корпуса.
    
    Args:
        text_size: Размер текста в байтах
        
    Returns:
        Рекомендуемый размер словаря
    """
    if text_size < 10_000:
        return 500
    elif text_size < 100_000:
        return 2_000
    elif text_size < 1_000_000:
        return 8_000
    elif text_size < 10_000_000:
        return 16_000
    elif text_size < 100_000_000:
        return 32_000
    else:
        return 50_000

def validate_corpus(file_path: Path) -> Tuple[bool, List[str]]:
    """
    Валидировать корпус перед обучением.
    
    Args:
        file_path: Путь к файлу корпуса
        
    Returns:
        (is_valid, issues) - кортеж валидности и списка проблем
    """
    issues = []
    
    if not file_path.exists():
        issues.append(f"Файл не существует: {file_path}")
        return False, issues
    
    file_size = file_path.stat().st_size
    if file_size == 0:
        issues.append("Файл пуст")
        return False, issues
    
    if file_size < 1_000:
        issues.append(f"Файл слишком маленький ({file_size} байт). Минимум 1 KB.")
    
    # Проверка кодировки
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = [f.readline() for _ in range(10)]
        
        if not any(first_lines):
            issues.append("Не удалось прочитать строки из файла")
    except UnicodeDecodeError:
        issues.append("Ошибка кодировки UTF-8. Конвертируйте файл.")
        return False, issues
    except Exception as e:
        issues.append(f"Ошибка чтения файла: {e}")
        return False, issues
    
    return len(issues) == 0, issues

def prepare_corpus(
    input_path: str,
    output_path: str = "corpus_cleaned.txt",
    min_line_length: int = 5,
    aggressive_clean: bool = False
) -> Dict:
    """
    Подготовить и очистить корпус для обучения токенизатора.
    
    Args:
        input_path: Путь к исходному файлу
        output_path: Путь для сохранения очищенного корпуса
        min_line_length: Минимальная длина строки
        aggressive_clean: Использовать агрессивную очистку
        
    Returns:
        Статистика обработки
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    # Валидация
    is_valid, issues = validate_corpus(input_file)
    if not is_valid:
        logger.error("❌ Ошибки корпуса:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return {'error': issues}
    
    logger.info(f"📂 Подготовка корпуса: {input_file}")
    logger.info(f"   Размер: {input_file.stat().st_size / 1_000_000:.2f} MB")
    
    cleaner = TextCleaner(aggressive=aggressive_clean)
    
    try:
        with open(input_file, 'r', encoding='utf-8', errors='replace') as f_in:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for line_idx, line in enumerate(f_in):
                    cleaner.stats['lines_processed'] += 1
                    
                    cleaned = cleaner.clean_text(line)
                    
                    if cleaner.is_valid_line(cleaned, min_line_length):
                        f_out.write(cleaned + '\n')
                        cleaner.stats['lines_cleaned'] += 1
                    
                    if (line_idx + 1) % 10000 == 0:
                        logger.debug(f"Обработано {line_idx + 1} строк...")
        
        # Финальные статистики
        input_size = input_file.stat().st_size
        output_size = output_file.stat().st_size
        stats = cleaner.get_stats()
        stats['input_size'] = input_size
        stats['output_size'] = output_size
        stats['compression_ratio'] = output_size / input_size if input_size > 0 else 0
        
        logger.info(f"✅ Корпус подготовлен:")
        logger.info(f"   Строк обработано: {stats['lines_processed']}")
        logger.info(f"   Строк сохранено: {stats['lines_cleaned']}")
        logger.info(f"   Коэффициент сжатия: {stats['compression_ratio']:.2%}")
        
        return stats
    
    except Exception as e:
        logger.error(f"❌ Ошибка подготовки корпуса: {e}")
        return {'error': str(e)}

def train(
    data_path: str = "data/corpus.txt",
    model_prefix: str = "tokenizer",
    vocab_size: Optional[int] = None,
    character_coverage: float = 0.9995,
    user_defined_symbols: Optional[list] = None,
    model_type: str = "bpe",
    train_params: Optional[Dict] = None
) -> bool:
    """
    Обучает токенизатор SentencePiece на корпусе.
    
    Args:
        data_path: Путь к файлу корпуса
        model_prefix: Префикс для сохранения моделей
        vocab_size: Размер словаря (если None, автоматически)
        character_coverage: Покрытие символов (по умолчанию 0.9995)
        user_defined_symbols: Определяемые пользователем символы
        model_type: Тип модели ('bpe' или 'unigram')
        train_params: Дополнительные параметры обучения
        
    Returns:
        True если успешно
    """
    data_file = Path(data_path)
    
    logger.info("🚀 Инициализация обучения токенизатора...")
    
    # Валидация корпуса
    is_valid, issues = validate_corpus(data_file)
    if not is_valid:
        logger.error("❌ Ошибки корпуса:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        logger.error(f"❌ Ошибка чтения файла: {e}")
        return False
    
    if not text.strip():
        logger.error("❌ Корпус пуст после обработки")
        return False
    
    #估算 размер словаря
    if vocab_size is None:
        vocab_size = estimate_vocab_size(len(text.encode('utf-8')))
    
    logger.info(f"📊 Статистика корпуса:")
    logger.info(f"   Размер: {len(text) / 1_000_000:.2f} MB ({len(text.encode('utf-8')) / 1_000_000:.2f} MB в bytes)")
    logger.info(f"   Строк: {len(text.splitlines())}")
    logger.info(f"   Уникальные символы: {len(set(text))}")
    logger.info(f"   Размер словаря: {vocab_size}")
    
    # Параметры обучения по умолчанию
    default_params = {
        "input": str(data_file),
        "model_prefix": model_prefix,
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "model_type": model_type,
        "pad_id": 0,
        "bos_id": 1,
        "eos_id": 2,
        "unk_id": 3,
        "hard_vocab_limit": False,
        "user_defined_symbols": user_defined_symbols or ["<newline>", "<tab>", "<url>", "<email>"],
        "normalization_rule_name": "nmt_nfkc",
        "remove_extra_whitespaces": True,
        "split_digits": False,
        "split_by_unicode_script": True,
        "split_by_whitespace": True,
        "treat_whitespace_as_suffix": False,
        "byte_fallback": True,  # Резервное решение для неизвестных символов
        "max_sentencepiece_length": 16,
    }
    
    # Объединить с переданными параметрами
    if train_params:
        default_params.update(train_params)
    
    logger.info(f"⚙️ Параметры обучения:")
    logger.info(f"   Тип модели: {default_params['model_type']}")
    logger.info(f"   Покрытие символов: {default_params['character_coverage']}")
    logger.info(f"   Специальные символы: {default_params['user_defined_symbols']}")
    
    try:
        logger.info("🔧 Обучение SentencePiece модели...")
        spm.SentencePieceTrainer.train(**default_params)
        logger.info("✅ Обучение завершено успешно")
    except Exception as e:
        logger.error(f"❌ Ошибка обучения: {e}")
        return False
    
    # Сохранить метаданные
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "model_type": model_type,
        "input_file": str(data_file),
        "model_prefix": model_prefix,
        "corpus_size_bytes": len(text.encode('utf-8')),
        "corpus_size_chars": len(text),
        "corpus_lines": len(text.splitlines()),
        "unique_chars": len(set(text)),
        "user_defined_symbols": default_params.get("user_defined_symbols", [])
    }
    
    try:
        metadata_path = f"{model_prefix}.meta.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Метаданные сохранены: {metadata_path}")
    except Exception as e:
        logger.error(f"⚠️ Ошибка сохранения метаданных: {e}")
    
    # Проверить созданные файлы
    model_file = Path(f"{model_prefix}.model")
    vocab_file = Path(f"{model_prefix}.vocab")
    
    if model_file.exists() and vocab_file.exists():
        logger.info(f"✅ Файлы модели созданы:")
        logger.info(f"   {model_file.name} ({model_file.stat().st_size / 1_000:.1f} KB)")
        logger.info(f"   {vocab_file.name} ({vocab_file.stat().st_size / 1_000:.1f} KB)")
        return True
    else:
        logger.error("❌ Файлы модели не созданы")
        return False

def benchmark_tokenizer(model_prefix: str, test_texts: Optional[List[str]] = None) -> Dict:
    """
    Протестировать обученный токенизатор.
    
    Args:
        model_prefix: Префикс модели
        test_texts: Тестовые тексты (если None, использовать по умолчанию)
        
    Returns:
        Результаты тестирования
    """
    model_file = Path(f"{model_prefix}.model")
    
    if not model_file.exists():
        logger.error(f"❌ Модель не найдена: {model_file}")
        return {'error': 'Model not found'}
    
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(str(model_file))
        
        if test_texts is None:
            test_texts = [
                "Привет, это тест токенизатора SentencePiece!",
                "Машинное обучение - это мощный инструмент.",
                "Как вас зовут? Меня зовут KINT.",
                "12345 и Some English words здесь."
            ]
        
        results = {
            'vocab_size': sp.get_piece_size(),
            'tests': []
        }
        
        logger.info("🧪 Тестирование токенизатора:")
        
        for text in test_texts:
            tokens = sp.encode(text)
            decoded = sp.decode(tokens)
            
            test_result = {
                'text': text,
                'num_tokens': len(tokens),
                'tokens': sp.encode(text, out_type=str)[:10],  # Первые 10
                'decoded': decoded,
                'compression_ratio': len(text) / len(tokens) if tokens else 0
            }
            results['tests'].append(test_result)
            
            logger.info(f"   Текст: '{text}'")
            logger.info(f"   Токенов: {len(tokens)}, Сжатие: {test_result['compression_ratio']:.2f}x")
        
        return results
    
    except Exception as e:
        logger.error(f"❌ Ошибка при тестировании: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # === ПОЛНЫЙ КОНВЕЙЕР ОБУЧЕНИЯ ===
    
    logger.info("=" * 60)
    logger.info("ОБУЧЕНИЕ ТОКЕНИЗАТОРА KINT")
    logger.info("=" * 60)
    
    # Этап 1: Подготовка корпуса
    logger.info("\n📝 Этап 1: Подготовка корпуса")
    prep_stats = prepare_corpus(
        input_path="data/corpus.txt",
        output_path="corpus_cleaned.txt",
        min_line_length=5,
        aggressive_clean=False
    )
    
    if 'error' in prep_stats:
        logger.error("❌ Ошибка подготовки корпуса")
        exit(1)
    
    # Этап 2: Обучение токенизатора
    logger.info("\n🤖 Этап 2: Обучение токенизатора")
    success = train(
        data_path="corpus_cleaned.txt",
        model_prefix="tokenizer",
        vocab_size=None,  # Автоматически
        character_coverage=0.9995,
        user_defined_symbols=["<newline>", "<tab>", "<url>", "<email>", "<code>", "<formula>"],
        model_type="bpe"
    )
    
    if not success:
        logger.error("❌ Ошибка обучения токенизатора")
        exit(1)
    
    # Этап 3: Тестирование
    logger.info("\n✅ Этап 3: Тестирование токенизатора")
    benchmark_results = benchmark_tokenizer("tokenizer")
    
    if 'error' not in benchmark_results:
        logger.info("\n✨ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        logger.info(f"   Словарь: {benchmark_results['vocab_size']} токенов")
    else:
        logger.error("\n❌ ОШИБКА ПРИ ТЕСТИРОВАНИИ")
