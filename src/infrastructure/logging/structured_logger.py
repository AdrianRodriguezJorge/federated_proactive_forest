"""Structured logging utilities."""
import logging
import json
from datetime import datetime
from typing import Dict, Any


class StructuredLogger:
    """
    Logger that outputs structured JSON logs.
    """

    def __init__(self, name: str = "federated_forest", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Add JSON handler
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)

    def log_experiment_start(self, config: Dict[str, Any]) -> None:
        """Log experiment start."""
        self.logger.info("Experiment started", extra={
            'event': 'experiment_start',
            'config': config,
            'timestamp': datetime.utcnow().isoformat()
        })

    def log_round_complete(self, round_num: int, metrics: Dict[str, Any]) -> None:
        """Log federated round completion."""
        self.logger.info("Round completed", extra={
            'event': 'round_complete',
            'round': round_num,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        })

    def log_error(self, error: Exception, context: str = "") -> None:
        """Log error."""
        self.logger.error("Error occurred", extra={
            'event': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.utcnow().isoformat()
        })


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    """

    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }

        # Add extra fields
        if hasattr(record, 'event'):
            log_entry['event'] = record.event
        if hasattr(record, 'config'):
            log_entry['config'] = record.config
        if hasattr(record, 'metrics'):
            log_entry['metrics'] = record.metrics
        if hasattr(record, 'error_type'):
            log_entry['error_type'] = record.error_type
        if hasattr(record, 'error_message'):
            log_entry['error_message'] = record.error_message
        if hasattr(record, 'context'):
            log_entry['context'] = record.context

        return json.dumps(log_entry)