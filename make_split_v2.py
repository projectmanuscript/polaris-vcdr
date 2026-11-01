import json, re, random, collections, os

s = json.load(open("project_split.json"))
excl = (
    json.load(open("exclusion_registry.json")).get("misregistered_images", [])
    if os.path.exists("exclusion_registry.json")
    else []
)
rng = random.Random(2026)


def source(k):
    ds, case = k.split("/", 1)
    if ds != "RIGA":
        return ds
    m = re.match(r"[A-Za-z]+", case)
    return "RIGA:" + (m.group(0) if m else "?")


groups = collections.defaultdict(list)
for k in s["train"]:
    groups[source(k)].append(k)
out = {"train": [], "val": [], "calib": [], "test": list(s["test"])}
for src, keys in sorted(groups.items()):
    keys = sorted(keys)
    rng.shuffle(keys)
    if src.startswith("RIGA"):
        n = len(keys)
        nv, nc = round(0.10 * n), round(0.20 * n)
        out["val"] += keys[:nv]
        out["calib"] += keys[nv : nv + nc]
        out["train"] += keys[nv + nc :]
    else:
        out["train"] += keys
sets = {k: set(v) for k, v in out.items()}
assert all(not (sets[a] & sets[b]) for a in sets for b in sets if a < b), "overlap between splits"
assert sets["train"] | sets["val"] | sets["calib"] == set(s["train"]), "lost or added training images"
assert out["test"] == s["test"], "test split changed"
print("Example keys:", s["train"][:3])
print("Misregistered entries in registry:", excl)
for name, keys in out.items():
    c = collections.Counter(source(k) for k in keys)
    print(f"{name:6s} {len(keys):4d}  " + "  ".join(f"{a}={b}" for a, b in sorted(c.items())))
json.dump(out, open("project_split_v2.json", "w"), indent=1)
print("Saved", os.path.abspath("project_split_v2.json"))
