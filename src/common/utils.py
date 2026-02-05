import yaml

def load_yaml(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data