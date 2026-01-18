import logging

class LogUtil:
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

def print_green(msg):
    print(f"\033[92m{msg}\033[0m")

def print_red(msg):
    print(f"\033[91m{msg}\033[0m")

def print_yellow(msg):
    print(f"\033[93m{msg}\033[0m")
