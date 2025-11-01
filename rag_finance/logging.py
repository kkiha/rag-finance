import logging
from typing import Optional

def setup_logging(level: int = logging.INFO, fmt: Optional[str] = None) -> None:
    if fmt is None:
        fmt = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(level=level, format=fmt)
