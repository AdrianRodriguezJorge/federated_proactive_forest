"""Logging infrastructure configuration.

Provides a centralized, multi-handler logging utility that routes output
to both file and terminal streams with customizable formats.
"""

import logging
import os


def setup_project_logger(
    name: str, log_file: str = "project_debug.log"
) -> logging.Logger:
    """Configura un logger centralizado para el proyecto.

    Envía mensajes de nivel INFO o superior a la consola con un formato limpio,
    y mensajes de nivel DEBUG o superior a un archivo con detalles completos
    (timestamp, nombre del módulo, nivel).

    Args:
        name (str): Nombre identificador del logger (normalmente __name__).
        log_file (str): Nombre del archivo donde se guardarán los logs.
            Defaults to "project_debug.log".

    Returns:
        logging.Logger: Instancia configurada del logger.
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
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Formato para la consola (más limpio para el usuario)
    console_formatter = logging.Formatter("%(message)s")

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
