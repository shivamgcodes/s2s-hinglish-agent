"""make_config.py TEMPLATE.yaml OUT.yaml key.sub=value ...  (values parsed as YAML scalars)"""
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
for kv in sys.argv[3:]:
    k, v = kv.split("=", 1)
    d = cfg
    *path, last = k.split(".")
    for p in path:
        d = d.setdefault(p, {})
    d[last] = yaml.safe_load(v)
yaml.safe_dump(cfg, open(sys.argv[2], "w"), sort_keys=False)
print(f"config -> {sys.argv[2]}")
