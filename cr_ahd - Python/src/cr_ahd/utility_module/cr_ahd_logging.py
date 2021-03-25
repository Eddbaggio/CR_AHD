import logging
from typing import Union
import os


def add_file_handler(logger: logging.Logger, path: Union[str, bytes, os.PathLike], mode='w', level=logging.DEBUG,
                     log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    f_handler = logging.FileHandler(path, mode)
    f_format = logging.Formatter(log_format)
    f_handler.setFormatter(f_format)
    f_handler.setLevel(level)
    logger.addHandler(f_handler)
    pass


def remove_all_file_handlers(logger: logging.Logger):
    handler: logging.Handler
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    pass


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
            'filename': '../../../data/Output/cr_ahd_log.log',
            'mode': 'w',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console'],
            'level': 'DEBUG',
            # 'propagate': False
        },
        # 'src.cr_ahd.solving_module.solver': {
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
