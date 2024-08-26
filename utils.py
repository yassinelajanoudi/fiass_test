import yaml
from box import Box  # Python dictionaries with advanced dot notation access.



def load_config():
    with open(r"./config.yml", "r", encoding="utf8") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
        return cfg
