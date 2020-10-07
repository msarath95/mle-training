import logging
import logging.config

import yaml


def read_config(path):
    """
    Loading a config file from path

    Parameters
    ----------
        path: str
            config path

    Return
    ------
        cfg: dict
            configurations
    """
    with open(path, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    return cfg


def configure_logger(log_conf="./config/log.conf", lvl="INFO"):
    """This function is to configure the logger.

    Parameters
    ----------
        log_conf: str, default ./config/log.conf
            log config path
        lvl: str, default "INFO"
            log messages level {'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG'}
    """
    logging.config.fileConfig(fname=log_conf)
    logger = logging.getLogger()
    logger.setLevel(lvl)
