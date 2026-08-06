# encoding=utf-8

import time
import traceback
from functools import wraps

import traceback
import time
import logging
from functools import wraps
from typing import Any, Callable, Optional


def exception_recorder(
    path: str = "exceptions.log",
    logger: Optional[logging.Logger] = None,
    log_to_console: bool = True,
    include_args: bool = False
) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                error_msg = [
                    f"{'='*50}",
                    f"     time: {timestamp}",
                    f" function: {func.__name__}",
                    f"     fine: {func.__code__.co_filename}:{func.__code__.co_firstlineno}",
                    f"expection: {type(e).__name__}: {str(e)}"
                ]
                
                if include_args:
                    if args:
                        error_msg.append(f"位置参数: {args}")
                    if kwargs:
                        error_msg.append(f"关键字参数: {kwargs}")
                
                error_msg.append("traceback:")
                error_msg.extend(traceback.format_exc().splitlines())
                error_msg.append(f"{'='*50}")
                
                full_error_message = "\n".join(error_msg) + "\n\n"

                if logger:
                    logger.error(full_error_message)
                else:
                    try:
                        with open(path, 'a', encoding='utf-8') as f:
                            f.write(full_error_message)
                    except Exception as write_error:
                        print(f"无法写入异常日志文件: {write_error}")
                        print(full_error_message)
                

                if log_to_console:
                    print(full_error_message)
                raise
        
        return wrapper
    return decorator
