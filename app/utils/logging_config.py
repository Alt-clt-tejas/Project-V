# app/utils/logging_config.py
import logging
import sys
from datetime import datetime
from pathlib import Path

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with key=value output."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present in the log call
        extra_fields = {k: v for k, v in record.__dict__.items() if k not in logging.LogRecord.__dict__ and k not in log_entry}
        if extra_fields:
            log_entry.update(extra_fields)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        formatted_pairs = []
        for key, value in log_entry.items():
            if isinstance(value, str) and ' ' in value:
                formatted_pairs.append(f'{key}="{value}"')
            else:
                formatted_pairs.append(f'{key}={value}')
        
        return ' '.join(formatted_pairs)

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Configure application-wide logging with structured output."""
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.root.setLevel(getattr(logging, log_level.upper()))
    logging.root.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter())
    logging.root.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(StructuredFormatter())
        logging.root.addHandler(file_handler)
    
    # Silence overly verbose third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("instagrapi").setLevel(logging.WARNING)
    
    logging.info("Logging configuration initialized", extra={"log_level": log_level})