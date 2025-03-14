import logging
import sys
from collections import defaultdict
import os
import time
from rich.logging import RichHandler


class WarningErrorCollectorHandler(logging.Handler):
    """
    A custom logging handler that collects warnings and errors in memory.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collected_logs = defaultdict(
            list
        )  # Use defaultdict to store logs by level

    def emit(self, record):
        try:
            msg = self.format(record)
            if record.levelno >= logging.WARNING:
                self.collected_logs[record.levelname].append(msg)
        except Exception:
            self.handleError(record)

    def get_collected_logs(self):
        return self.collected_logs

    def clear_collected_logs(self):
        self.collected_logs = defaultdict(list)


def setup_logging(log_level=logging.INFO):
    """
    Configures logging for the application, including console output, file logging,
    and a custom handler for collecting warnings and errors.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(
        logging.DEBUG
    )  # Set root logger to DEBUG to capture all levels

    # Modified Formatter to include traceback info
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s\n%(exc_text)s"  # Added %(exc_text)s
    )

    # # Console handler
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setLevel(log_level)  # Set console handler to specified log level
    # console_handler.setFormatter(formatter)
    # root_logger.addHandler(console_handler)

    # Rich handler for console logging
    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    rich_handler.setLevel(log_level)
    rich_handler.setFormatter(formatter)
    root_logger.addHandler(rich_handler)
    # File handler (optional - you can configure if you need file logging)
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs("logs", exist_ok=True)
    os.makedirs(f"logs/{run_timestamp}", exist_ok=True)
    file_handler = logging.FileHandler(f"logs/{run_timestamp}/{__name__}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Custom handler to collect warnings and errors
    collector_handler = WarningErrorCollectorHandler()
    collector_handler.setLevel(
        logging.WARNING
    )  # Collector starts collecting from WARNING level
    collector_handler.setFormatter(formatter)  # Use the same formatter
    root_logger.addHandler(collector_handler)

    return root_logger, collector_handler  # Return both logger and handler for access


def get_logger(name=None):
    """
    Retrieves a logger instance. If name is None, it returns the root logger.
    """
    if name:
        return logging.getLogger(name)
    else:
        return logging.getLogger()  # Returns root logger if no name is provided


if __name__ == "__main__":
    # Example Usage
    root_logger, collector_handler = setup_logging(log_level=logging.INFO)
    logger = get_logger(__name__)

    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    try:
        raise ValueError("Example ValueError for traceback logging")
    except ValueError as e:
        logger.error(
            "This is an error message with traceback", exc_info=True
        )  # Pass exc_info=True
    logger.critical("This is a critical message.")

    collected_logs = collector_handler.get_collected_logs()

    print("\n--- Collected Warnings and Errors Summary ---")
    for level, logs in collected_logs.items():
        if logs:  # Only print if there are logs for this level
            print(f"\n--- {level} Logs ---")
            for log_message in logs:
                print(log_message)
