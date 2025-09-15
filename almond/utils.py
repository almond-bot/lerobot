import logging

from env import LOGTAIL_HOST, LOGTAIL_TOKEN
from logtail import LogtailHandler

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def create_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False

    std_handler = logging.StreamHandler()
    formatter = logging.Formatter(LOG_FORMAT)
    std_handler.setFormatter(formatter)
    logger.addHandler(std_handler)

    handler = LogtailHandler(source_token=LOGTAIL_TOKEN, host=LOGTAIL_HOST)
    logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)

    return logger
