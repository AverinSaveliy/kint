"""
Главный точка входа для KINT МЕГАИНТЕЛЛЕКТ
"""
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def main():
    """Главная функция"""
    try:
        from interfaces.cli import run_cli
        run_cli()
    except KeyboardInterrupt:
        print("\n🌟 Выход...")
        sys.exit(0)
    except Exception as e:
        logging.error(f"❌ Ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
