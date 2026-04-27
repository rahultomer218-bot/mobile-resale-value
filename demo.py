from mobile_resale_value.logger import logger
from mobile_resale_value.exception import MobileResaleException
import sys

# Test logger
logger.info("Logger is working!")
logger.warning("This is a warning!")
logger.error("This is an error!")
logger.critical("This is a critical message!")
print("Logger test completed! Check the logs/ folder for the log file.")

# Test exception
try:
    a = 1 / 0
except Exception as e:
    raise MobileResaleException(e, sys)