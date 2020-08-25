import yaml


def read_config(path):
    """
    Loading a config file from path
    Parameters:
        path (string): config path
    Returns:
        cfg (dict): configurations
    """
    with open(path, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    return cfg
