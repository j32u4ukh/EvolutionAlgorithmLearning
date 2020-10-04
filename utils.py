import inspect
import logging


def getFuncName():
    """
    :return: 當前函式名稱
    """
    return inspect.stack()[1][3]


def getLogger(logger_name, logger_level=logging.DEBUG, logger_format=None):
    if logger_format is None:
        logger_format = '%(asctime)s %(levelname)s: [%(name)s] %(funcName)s | ' \
                        '%(message)s (line: %(lineno)d, %(pathname)s)'

    # logger.debug("debug")
    # logger.info("info")
    # logger.warning("warning")
    # logger.error("error")
    # logger.critical("critical")

    # logger_name: 設置 logger 名稱
    logger = logging.getLogger(logger_name)

    # 避免重複輸出 log
    if not logger.handlers:
        # 設置 logger 的 level
        logger.setLevel(logger_level)

        # 創建一個輸出日誌到控制台的 StreamHandler
        handler = logging.StreamHandler()

        # 設置 logger 的格式
        formatter = logging.Formatter(logger_format)

        # 將格式添加給 StreamHandler
        handler.setFormatter(formatter)

        # 將 handler 添加給 logger
        logger.addHandler(handler)

    return logger
