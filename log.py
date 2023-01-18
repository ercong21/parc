# -*- coding: utf-8 -*-
# Author: Niel
# Date: 2022/6/12  23:56

"""
This file contains basic logging logic.
"""

import logging

output_path = 'log'

names = set()

def __setup_custom_logger(name: str) -> logging.Logger:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    names.add(name)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    fileHandler = logging.FileHandler(output_path)
    fileHandler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
    return logger

def get_logger(name: str) -> logging.Logger:
    if name in names:
        return logging.getLogger(name)
    else:
        return __setup_custom_logger(name)