import logging


class Logger:
    @classmethod
    def get_logger(
        self, logger_name, filename: str, level=logging.INFO
    ) -> logging.Logger:
        # set file handler for logger
        # that redirect log info to file
        handler = logging.FileHandler(filename, mode="w")

        # create logger with specified file
        specified_logger = logging.getLogger(logger_name)
        specified_logger.setLevel(level)
        specified_logger.addHandler(handler)
        specified_logger.propagate = False

        return specified_logger
