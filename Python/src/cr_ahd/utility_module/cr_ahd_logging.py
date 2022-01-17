import logging
import os
import pathlib
from typing import Union

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'brief': {
            'format': '[%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'brief',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
        'info_file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': './data/Output/logs/cr_ahd_log.log',
            'mode': 'w',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console'],
            'level': 'DEBUG',
            # 'propagate': False
        },
        # 'src.cr_ahd.routing_module.solver': {
        #     'handlers': ['file'],
        #     'level': 'DEBUG',
        # },
        # 'src.cr_ahd.core_module.tour': {
        #     'handlers': ['file'],
        #     'level': 'DEBUG',
        # },
        # '__main__': {  # if __name__ == '__main__'
        #     'handlers': ['file'],
        #     'level': 'DEBUG',
        # },
    }
}

SUCCESS = 21
logging.addLevelName(SUCCESS, 'SUCC')


class CustomFormatter(logging.Formatter):
    """
    https://stackoverflow.com/a/56944256/15467861
    """
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = LOGGING_CONFIG['formatters']['brief']['format']
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
        SUCCESS: green + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def add_handlers(logger: logging.Logger, path: Union[str, bytes, os.PathLike], mode='w', level=logging.DEBUG,
                 log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    f_handler = logging.FileHandler(path, mode)
    f_format = logging.Formatter(log_format)
    f_handler.setFormatter(f_format)
    f_handler.setLevel(level)
    logger.addHandler(f_handler)

    pass


def remove_all_handlers(logger: logging.Logger):
    handler: logging.Handler
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    pass
