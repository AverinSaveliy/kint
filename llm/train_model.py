# ================================================================
# train.py для KINT (LLM с локальной квантовой компонентой)
# ================================================================

import torch, time, json, tempfile, logging
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from llm.model import KINTLanguageModel
from llm.tokenizer import RussianBPETokenizer
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import numpy as np
from collections import defaultdict

# ================================================================
# КОНФИГУРАЦИЯ И ПАРАМЕТРЫ
# ================================================================

MAX_EPOCHS = 50
BATCH_SIZE = 8
BLOCK_SIZE = 256
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5
PATIENCE = 5
GRAD_CLIP = 1.0

LOSS_WEIGHTS = {
    "lm": 1.0,
    "quantum": 0.1,
    "contrastive": 0.05
}

EPOCH_STATE_FILE = Path("epochs/epoch_state.json")
BEST_MODEL_PATH = Path("epochs/best_model.pth")
LOG_DIR = Path("epochs/logs")
LOG_FILE = LOG_DIR / f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

EPOCH_DIR = Path("epochs/saved_epochs")
EPOCH_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================================================
# DATASET С УЛУЧШЕННОЙ АУГМЕНТАЦИЕЙ
# ================================================================

class TextDataset(Dataset):
    """Dataset с поддержкой аугментации и кэширования"""
    def __init__(self, file_path: str, tokenizer: RussianBPETokenizer, block_size: int, 
                 augment: bool = True, cache_size: int = 10000):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.augment = augment
        self.cache_size = cache_size

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = tokenizer.encode(text, out_type="int")
        self.examples = []
        self.token_counts = defaultdict(int)
        
        for i in range(0, len(tokens) - block_size, block_size):
            block = tokens[i:i + block_size]
            example = torch.tensor(block, dtype=torch.long)
            self.examples.append(example)
            
            # Статистика токенов
            for token in block:
                self.token_counts[token] += 1

        if augment:
            self._augment_data()

        logger.info(f"Загружено {len(self.examples)} примеров")
        logger.info(f"Уникальные токены: {len(self.token_counts)}")

    def _augment_data(self):
        """Аугментация данных через варианты блоков"""
        original_len = len(self.examples)
        for i in range(min(original_len, self.cache_size)):
            example = self.examples[i]
            # Перестановка
            shuffled = example[torch.randperm(len(example))]
            self.examples.append(shuffled)
            # Обратное направление
            reversed_ex = example.flip(0)
            self.examples.append(reversed_ex)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        input_ids = example[:-1]
        labels = example[1:]
        return {"input_ids": input_ids, "labels": labels}

# ================================================================
# МЕТРИКИ И МОНИТОРИНГ
# ================================================================

class TrainingMetrics:
    """Отслеживание метрик обучения"""
    def __init__(self):
        self.history = defaultdict(list)
        self.best_metrics = {}
        
    def update(self, metrics: Dict[str, float], epoch: int):
        """Обновить метрики"""
        for key, value in metrics.items():
            self.history[key].append((epoch, value))
            
        # Обновить лучшие значения
        for key, value in metrics.items():
            if key not in self.best_metrics or value < self.best_metrics[key]:
                self.best_metrics[key] = value
    
    def get_summary(self) -> str:
        """Получить сводку лучших метрик"""
        lines = ["=== ЛУЧШИЕ МЕТРИКИ ==="]
        for key, value in self.best_metrics.items():
            lines.append(f"{key}: {value:.6f}")
        return "\n".join(lines)

class ProgressTracker:
    """Отслеживание прогресса обучения"""
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.epoch_times = []
    
    def log_epoch(self, epoch: int, loss: float, lr: float):
        """Логирование эпохи"""
        elapsed = time.time() - self.start_time
        self.epoch_times.append(elapsed)
        
        avg_time = np.mean(self.epoch_times[-10:]) if len(self.epoch_times) > 0 else 0
        remaining = avg_time * (self.total_epochs - epoch - 1)
        
        hours, remainder = divmod(remaining, 3600)
        minutes, _ = divmod(remainder, 60)
        
        logger.info(
            f"Epoch {epoch+1}/{self.total_epochs} | Loss: {loss:.6f} | "
            f"LR: {lr:.2e} | ETA: {int(hours)}h {int(minutes)}m"
        )

# ================================================================
# СОХРАНЕНИЕ И ВОССТАНОВЛЕНИЕ
# ================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    patience_counter: int,
    metrics: TrainingMetrics,
    checkpoint_path: Path,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    """Сохранить контрольную точку"""
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_loss": best_loss,
        "patience_counter": patience_counter,
        "metrics": dict(metrics.history),
        "timestamp": datetime.now().isoformat(),
        "learning_rate": optimizer.param_groups[0]["lr"]
    }
    
    if scheduler:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            torch.save(checkpoint, tmp.name)
            Path(tmp.name).replace(checkpoint_path)
        logger.info(f"✅ Контрольная точка сохранена: {checkpoint_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Не удалось сохранить контрольную точку: {e}")
        return False

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Tuple[int, float, int, TrainingMetrics]:
    """Загрузить контрольную точку"""
    if not checkpoint_path.exists():
        logger.info("Контрольная точка не найдена. Начинаем с нуля.")
        return 0, float("inf"), 0, TrainingMetrics()
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint["model_state"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        if scheduler and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        metrics = TrainingMetrics()
        if "metrics" in checkpoint:
            metrics.history = defaultdict(list, checkpoint["metrics"])
        
        logger.info(f"✅ Контрольная точка загружена из эпохи {checkpoint['epoch'] + 1}")
        return (
            checkpoint["epoch"] + 1,
            checkpoint.get("best_loss", float("inf")),
            checkpoint.get("patience_counter", 0),
            metrics
        )
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки контрольной точки: {e}")
        return 0, float("inf"), 0, TrainingMetrics()

# ================================================================
# ФУНКЦИИ ОБУЧЕНИЯ
# ================================================================

def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    hidden_states: Optional[torch.Tensor] = None,
    augmented_states: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Вычислить комбинированную потерю"""
    # Основная потеря языковой модели
    lm_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    )
    
    losses = {"lm": lm_loss.item()}
    total_loss = LOSS_WEIGHTS["lm"] * lm_loss
    
    # Квантовая регуляризация
    if hasattr(model, 'quantum_layer'):
        quantum_weights = model.quantum_layer.weights
        quantum_entanglers = model.quantum_layer.entanglers
        quantum_loss = (
            torch.norm(quantum_weights) ** 2 +
            torch.norm(quantum_entanglers) ** 2
        ) / (quantum_weights.numel() + quantum_entanglers.numel())
        
        losses["quantum"] = quantum_loss.item()
        total_loss = total_loss + LOSS_WEIGHTS["quantum"] * quantum_loss
    
    # Контрастивная потеря (если есть скрытые состояния)
    if hidden_states is not None and augmented_states is not None:
        try:
            contrastive_loss = model.compute_contrastive_loss(hidden_states, augmented_states)
            losses["contrastive"] = contrastive_loss.item()
            total_loss = total_loss + LOSS_WEIGHTS["contrastive"] * contrastive_loss
        except Exception as e:
            logger.debug(f"Контрастивная потеря пропущена: {e}")
    
    return total_loss, losses

def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Обучить одну эпоху"""
    model.train()
    epoch_loss = 0.0
    loss_components = defaultdict(float)
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        # Прямой проход
        output = model(input_ids, return_logits=True, enable_reasoning=True)
        
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
        
        # Вычислить потерю
        loss, loss_dict = compute_loss(logits, labels, model)
        
        # Проверка на NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Loss содержит NaN/Inf на батче {batch_idx}, пропускаем")
            continue
        
        # Обратный проход
        loss.backward()
        
        # Градиентный клиппинг
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        # Накопление статистики
        epoch_loss += loss.item()
        for key, value in loss_dict.items():
            loss_components[key] += value
        
        # Логирование батча (каждые 50)
        if batch_idx % 50 == 0:
            avg_loss = epoch_loss / (batch_idx + 1)
            logger.debug(f"  Batch {batch_idx} | Loss: {avg_loss:.6f}")
    
    # Средние значения за эпоху
    avg_loss = epoch_loss / len(dataloader)
    metrics = {"loss": avg_loss}
    
    for key, value in loss_components.items():
        metrics[f"{key}_loss"] = value / len(dataloader)
    
    return metrics

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Оценить модель на валидационном наборе"""
    model.eval()
    total_loss = 0.0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        output = model(input_ids, return_logits=True)
        logits = output['logits'] if isinstance(output, dict) else output
        
        loss, _ = compute_loss(logits, labels, model)
        
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
    
    return {"eval_loss": total_loss / len(dataloader)}

# ================================================================
# ОСНОВНОЙ ЦИКЛ ОБУЧЕНИЯ
# ================================================================

def train():
    """Основной цикл обучения"""
    global MAX_EPOCHS
    
    logger.info("🚀 Начинаем МЕГАОБУЧЕНИЕ KINT...")
    
    # === ИНИЦИАЛИЗАЦИЯ ===
    tokenizer = RussianBPETokenizer()
    model = KINTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        dim=2048,
        depth=64,
        heads=64,
        quantum_qubits=32,
        num_reasoning_steps=50
    )
    
    # Выбор устройства
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("✅ GPU (Apple Metal) - MPS")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("✅ GPU (NVIDIA) - CUDA")
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ CPU (будет медленно)")
    
    model.to(device)
    
    # === ЗАГРУЗКА ДАННЫХ ===
    try:
        dataset = TextDataset("data/corpus.txt", tokenizer, BLOCK_SIZE, augment=True)
    except FileNotFoundError:
        logger.error("❌ Файл data/corpus.txt не найден!")
        return
    
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        pin_memory=(device.type == "cuda"),
        num_workers=0
    )
    
    logger.info(f"📊 Данные: {len(train_dataset)} обучение, {len(val_dataset)} валидация")
    
    # === ОПТИМИЗАТОР И SCHEDULER ===
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98)
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=MAX_EPOCHS * len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # === ВОССТАНОВЛЕНИЕ СОСТОЯНИЯ ===
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EPOCH_DIR.mkdir(parents=True, exist_ok=True)
    
    start_epoch, best_loss, patience_counter, metrics_tracker = load_checkpoint(
        model, optimizer, EPOCH_STATE_FILE, device, scheduler
    )
    
    progress_tracker = ProgressTracker(MAX_EPOCHS)
    training_metrics = TrainingMetrics()
    
    # === ОСНОВНОЙ ЦИКЛ ===
    for epoch in range(start_epoch, MAX_EPOCHS):
        try:
            # Обучение
            train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
            scheduler.step()
            
            # Валидация (каждые 5 эпох)
            if epoch % 5 == 0:
                val_metrics = evaluate(model, val_loader, device)
                train_metrics.update(val_metrics)
            
            # Обновить ме трики
            training_metrics.update(train_metrics, epoch)
            
            current_loss = train_metrics["loss"]
            current_lr = optimizer.param_groups[0]["lr"]
            
            # Логирование
            progress_tracker.log_epoch(epoch, current_loss, current_lr)
            
            # Проверка улучшения
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
                
                # Сохранить лучшую модель
                save_checkpoint(
                    model, optimizer, epoch, best_loss, 
                    patience_counter, training_metrics, BEST_MODEL_PATH, scheduler
                )
                logger.info(f"🎯 Новый лучший loss: {best_loss:.6f}")
            else:
                patience_counter += 1
                
                if patience_counter >= PATIENCE:
                    logger.info(f"⏹️ Ранняя остановка на эпохе {epoch+1}")
                    break
            
            # Периодическое сохранение
            if epoch % 10 == 0:
                epoch_checkpoint = EPOCH_DIR / f"model_epoch_{epoch+1}.pth"
                save_checkpoint(
                    model, optimizer, epoch, best_loss,
                    patience_counter, training_metrics, epoch_checkpoint, scheduler
                )
        
        except KeyboardInterrupt:
            logger.info("⏸️ Обучение прервано пользователем")
            break
        except Exception as e:
            logger.error(f"❌ Ошибка на эпохе {epoch+1}: {e}")
            continue
    
    # === ФИНАЛ ===
    logger.info("\n" + "="*60)
    logger.info("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    logger.info("="*60)
    logger.info(training_metrics.get_summary())
    logger.info(f"Лучшая модель: {BEST_MODEL_PATH}")
    logger.info(f"Логи: {LOG_FILE}")

if __name__ == "__main__":
    train()