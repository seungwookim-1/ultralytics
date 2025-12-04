import logging
from ultralytics.utils import LOGGER

class FreezeWarningFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self._already_logged = False

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "requires_grad=True' for frozen layer" in msg:
            if self._already_logged:
                return False
            self._already_logged = True
        return True

def install_freeze_warning_filter() -> None:
    """동일 freeze-warning이 여러 번 찍히는 것을 한 번으로 제한."""
    LOGGER.addFilter(FreezeWarningFilter())