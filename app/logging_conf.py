import os
from datetime import datetime

os.makedirs("logs", exist_ok=True)
# Get the current time for the log file name
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"logs/app_{current_time}.log"

log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailedFormatter": {
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "fileHandler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailedFormatter",
            "filename": log_filename,
            "mode": "w",
            "encoding": "utf-8",
        }
    },
    "loggers": {"": {"level": "DEBUG", "handlers": ["fileHandler"]}},  # Root logger
}
