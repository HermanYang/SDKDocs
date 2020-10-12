import enum
import logging


class LogLevel(enum.Enum):
    DEBUG = 0

    # Issue logic
    ISSUE = 1
    COMPLETE = 2
    HAZARDS = 3
    ISSUE_WINDOW = 4
    DEPENDENCIES = 5

    # Modeling
    ARCH_MODEL = 20
    INSTRUCTION_MODEL = 21


LEVELS_ENABLED = set([LogLevel.DEBUG])

PREFIXES = {
    LogLevel.DEBUG: "",
    LogLevel.ISSUE: "******",
    LogLevel.COMPLETE: "*-----*",
    LogLevel.HAZARDS: "\t*",
    LogLevel.DEPENDENCIES: "\t^",
    LogLevel.ISSUE_WINDOW: "@@ ",
    LogLevel.ARCH_MODEL: "% ",
    LogLevel.INSTRUCTION_MODEL: "$ "
}


def log(msg, level):
    if level in LEVELS_ENABLED:
        logging.info("{0}{1}".format(PREFIXES[level], msg))


def debug(msg):
    log(msg, LogLevel.DEBUG)
