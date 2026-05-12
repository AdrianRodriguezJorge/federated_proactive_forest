import logging
import os
from datetime import datetime

def setup_project_logger(name: str, log_file: str = "project_debug.log"):
    """
    Configura un logger centralizado para el proyecto.
    Envía INFO a la consola (limpio) y DEBUG al archivo (detallado).
    """
    log_dir = "results/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Evitar duplicar handlers si ya existen
    if logger.hasHandlers():
        return logger

    # Formato para el archivo (detallado con timestamp y nombre del módulo)
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Formato para la consola (más limpio para el usuario)
    console_formatter = logging.Formatter(
        '%(message)s'
    )

    # Handler para Archivo (DEBUG)
    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)

    # Handler para Consola (INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
