from contextlib import contextmanager
import torch, logging, sys, json, signal
from typing import List, Optional
from pathlib import Path
from llm.model import KINTLanguageModel
from llm.tokenizer import RussianBPETokenizer

# Путь к файлу конфигурации
CONFIG_FILE = Path("config.json")

def load_config() -> dict:
    """Загружает конфигурацию из JSON‑файла."""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        # Преобразуем строки в Path, если нужно
        if "MODEL_PATH" in config:
            config["MODEL_PATH"] = Path(config["MODEL_PATH"])

        # Добавляем значение по умолчанию для TOP_P
        if "TOP_P" not in config:
            config["TOP_P"] = None  # или другое значение по умолчанию
        
        return config
    except FileNotFoundError:
        print(f"⚠️ Файл конфигурации {CONFIG_FILE} не найден. Используются дефолтные значения.")
        return {
            "MAX_INPUT_LENGTH": 200,
            "MAX_CONTEXT_LENGTH": 1024,
            "MAX_NEW_TOKENS": 150,
            "DEFAULT_TEMPERATURE": 0.8,
            "DEFAULT_TOP_K": 50,
            "MODEL_PATH": Path("epochs/best_model.pth"),
            "TIMEOUT_SECONDS": 60,
            "TOP_P": None,
            "REPETITION_PENALTY": 1.2
        }
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка чтения JSON из {CONFIG_FILE}: {e}")
        sys.exit(1)

CONFIG = load_config()




# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("kint_cli.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)



# === ОБРАБОТКА ТАЙМАУТА ===
@contextmanager
def timeout(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutError(f"Превышено время генерации ({seconds} сек)")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)



# === СЕРВИСЫ ===
class TokenizerService:
    def __init__(self, tokenizer: RussianBPETokenizer):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        if not text.strip():
            raise ValueError("Пустой ввод")
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)



class GenerationService:
    def __init__(
        self,
        model: KINTLanguageModel,
        tokenizer_service: TokenizerService,
        device: str
    ):
        self.model = model.to(device).eval()
        self.tokenizer_service = tokenizer_service
        self.device = device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = CONFIG["MAX_NEW_TOKENS"],
        temperature: float = CONFIG["DEFAULT_TEMPERATURE"],
        top_k: int = CONFIG["DEFAULT_TOP_K"],
        top_p: Optional[float] = CONFIG["TOP_P"],
        repetition_penalty: float = CONFIG["REPETITION_PENALTY"]
    ) -> str:
        try:
            # Токенизация
            tokens = self.tokenizer_service.encode(prompt)
            if len(tokens) > CONFIG["MAX_INPUT_LENGTH"]:
                raise ValueError(
                    f"Превышение длины ввода ({len(tokens)} > {CONFIG['MAX_INPUT_LENGTH']})"
                )

            input_tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)

            with torch.no_grad(), timeout(CONFIG["TIMEOUT_SECONDS"]):
                for _ in range(max_new_tokens):
                    logits = self.model(input_tokens)
                    
                    # Проверяем, что logits является тензором
                    if isinstance(logits, list):
                        logits = torch.tensor(logits, device=self.device)
                    
                    logits = logits[:, -1, :] / temperature

                    # Топ‑k фильтрация
                    if top_k and top_k > 0:
                        v, _ = torch.topk(logits, top_k)
                        logits[logits < v[:, [-1]]] = -float('inf')

                    # Nucleus sampling (top‑p)
                    if top_p and top_p > 0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(
                            torch.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        filtered_indices = sorted_indices[:, cumulative_probs > top_p]
                        logits[:, filtered_indices] = -float('inf')

                    # Штраф за повторы
                    for token in set(input_tokens[0].tolist()):
                        logits[:, token] /= repetition_penalty

                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    input_tokens = torch.cat([input_tokens, next_token], dim=1)

                    if (next_token.item() == self.tokenizer_service.tokenizer.eos_id
                        or input_tokens.size(1) >= CONFIG["MAX_CONTEXT_LENGTH"]):
                        break

            generated_tokens = input_tokens[0].tolist()
            return self.tokenizer_service.decode(generated_tokens)

        except TimeoutError as e:
            logger.error(f"Таймаут генерации: {e}")
            return "❌ Превышено время генерации"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("Недостаточно видеопамяти")
                return "❌ Недостаточно видеопамяти. Попробуйте сократить длину ввода."
            else:
                logger.error(f"Ошибка генерации: {e}")
                return f"❌ Ошибка генерации: {e}"
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return f"❌ Неизвестная ошибка: {e}"

# === CLI ===
class KINTCLI:
    """KINT CLI с расширенными возможностями"""
    def __init__(self):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"  # Metal Performance Shaders
        else:
            self.device = "cpu"
        self.history: List[str] = []
        self.temperature = 0.6
        self.top_k = 100
        self.top_p = 0.98
        self.repetition_penalty = 1.1
        self.enable_reasoning = True
        self.enable_future_prediction = True
        self.reasoning_depth = 50
        
        logger.info("🧠 KINT МЕГАИНТЕЛЛЕКТ инициализирована")

    def load_model(self) -> Optional[GenerationService]:
        try:
            tokenizer = RussianBPETokenizer()
            
            # Загрузка МЕГАМОДЕЛИ
            model = KINTLanguageModel(
                vocab_size=tokenizer.vocab_size,
                dim=2048,
                depth=64,
                heads=64,
                quantum_qubits=32,
                num_reasoning_steps=50
            )

            if CONFIG["MODEL_PATH"].exists():
                state_dict = torch.load(
                    CONFIG["MODEL_PATH"],
                    map_location=self.device,
                    weights_only=True
                )
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"✅ МЕГАМОДЕЛЬ загружена: {CONFIG['MODEL_PATH']}")
            else:
                logger.warning("⚠️  МЕГАМОДЕЛЬ не найдена. Используется инициализированная.")

            model.eval()
            if self.device == "cuda":
                model = model.half()

            tokenizer_service = TokenizerService(tokenizer)
            return GenerationService(model, tokenizer_service, self.device)

        except Exception as e:
            logger.critical(f"❌ Ошибка загрузки МЕГАМОДЕЛИ: {e}")
            return None

    def show_help(self):
        print(
            "🚀 KINT МЕГАИНТЕЛЛЕКТ - Расширенные команды:\n"
            "- exit/quit: выход\n"
            "- help: справка\n"
            "- reason <глубина>: установить глубину рассуждений (1-50)\n"
            "- temp <значение>: температура (0.1-1.5)\n"
            "- prediction on/off: будущее предсказание\n"
            "- analyze <текст>: глубокий анализ текста\n"
            "- predict <контекст>: предсказать развитие\n"
            "- reset: сброс параметров"
        )

    def parse_command(self, user_input: str) -> bool:
        cmd = user_input.lower().strip()

        if cmd in {"exit", "quit"}:
            print("🌟 До свидания!")
            return False

        elif cmd == "help":
            self.show_help()
            return True

        elif cmd.startswith("reason "):
            try:
                value = int(cmd.split()[1])
                if 1 <= value <= 50:
                    self.reasoning_depth = value
                    print(f"✅ Глубина рассуждений: {self.reasoning_depth}")
                else:
                    print("❌ Значение от 1 до 50")
            except:
                print("❌ Пример: reason 30")
            return True

        elif cmd.startswith("prediction "):
            state = cmd.split()[1].lower()
            self.enable_future_prediction = state == "on"
            print(f"✅ Предсказание: {'ВКЛО' if self.enable_future_prediction else 'ВЫКЛО'}")
            return True

        elif cmd.startswith("analyze "):
            text = user_input[8:]
            print("🔍 Анализ...\n")
            # Здесь будет расширенный анализ
            print(f"📊 Анализ текста: '{text[:50]}...'")
            return True

        return False

    def _get_device_info(self) -> str:
        """Возвращает человекочитаемое описание устройства."""
        if self.device == "cuda":
            # NVIDIA GPU
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NVIDIA GPU"
            return f"✅ Модель на GPU (CUDA): {gpu_name}"
        elif self.device == "mps":
            # Apple Silicon (M1/M2/M3)
            return "✅ Модель на GPU (Apple MPS: Apple Silicon)"
        elif self.device == "cpu":
            # CPU (универсальный вариант)
            import platform
            cpu_info = platform.processor() or platform.machine()
            return f"⚠️ Модель на CPU ({cpu_info}): генерация может быть медленной"
        elif self.device.startswith("xla"):
            # TPU (Google Cloud)
            return "✅ Модель на TPU (Google XLA)"
        elif self.device.startswith("hpu"):
            # Habana Goya (Intel)
            return "✅ Модель на GPU (Habana Goya)"
        elif self.device.startswith("ort"):
            # ONNX Runtime
            return "✅ Модель на ускорителе (ONNX Runtime)"
        elif self.device.startswith("npu"):
            # Huawei Ascend
            return "✅ Модель на NPU (Huawei Ascend)"
        else:
            # Неизвестный/нестандартный бэкенд
            return f"ℹ️ Модель на устройстве: {self.device} (неизвестный тип)"


    def run(self):
        print("🚀 KINT МЕГАИНТЕЛЛЕКТ - СУПЕР ИИ")
        print("=" * 60)
        print("Введите 'help' для справки")
        print("=" * 60)

        generation_service = self.load_model()
        if not generation_service:
            sys.exit(1)

        print(self._get_device_info())

        while True:
            try:
                user_input = input("\n🧠 Вы> ").strip()
                if not user_input:
                    continue

                if self.parse_command(user_input):
                    continue

                print("⚡ KINT> ", end="", flush=True)
                response = generation_service.generate(
                    prompt=user_input,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty
                )
                print(response)

                self.history.append(f"Вы> {user_input}")
                self.history.append(f"KINT> {response}")

            except KeyboardInterrupt:
                print("\n🌟 До свидания!")
                break
            except Exception as e:
                logger.error(f"Ошибка: {e}")
                print(f"❌ Ошибка: {e}")

def run_cli():
    """Запустить CLI"""
    cli = KINTCLI()
    cli.run()
