import yaml
import pickle
import sys

with open(sys.argv[1]) as f:
    yaml_data = yaml.load(f, Loader=yaml.Loader)
    with open(sys.argv[1][:-4] + "pkl", "wb") as f2:
        pickle.dump(yaml_data, f2)
