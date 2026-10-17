# logging_config.py
import logging
from logging.config import dictConfig
​
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,  # 避免禁用第三方库日志
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        },
        "simple": {
            "format": "%(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.handlers.RotatingFileHandler", # 轮转文件
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,      # 保留 5 个备份
            "encoding": "utf-8"
        },
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple"
        }
    },
    "loggers": {
        "myapp": {  # 为你的应用定义 logger
            "level": "DEBUG",
            "handlers": ["file", "console"],
            "propagate": False
        }
    }
}
​
dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("myapp") # 创建全局 logger 实例