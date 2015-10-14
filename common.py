import logging


def setLogger():
  logging.Formatter(fmt='[%(asctime)s %(levelname)-5s %(name)s] %(message)s',datefmt='%H:%M:%S')
  logger = logging.getLogger("")
  log_console = logging.StreamHandler()
  # log_console.setFormatter(log_fmt)
  logger.setLevel(logging.DEBUG)
  logger.addHandler(log_console)
  return logger