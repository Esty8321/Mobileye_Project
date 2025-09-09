# update_project/scripts/make_classes_txt.py
from __future__ import annotations
import json, sys
from pathlib import Path

def load_classes_from_json(p: Path):
    obj = json.loads(p.read_text(encoding="utf-8"))
    # 1) ["healthy","sick", ...]
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        return obj
    # 2) {"classes": ["healthy","sick", ...]}
    if isinstance(obj, dict) and isinstance(obj.get("classes"), list):
        return [str(x) for x in obj["classes"]]
    # 3) {"healthy":0,"sick":1}  (map name->index)
    if isinstance(obj, dict) and all(isinstance(v, int) for v in obj.values()):
        pairs = sorted(((int(v), str(k)) for k, v in obj.items()))
        return [name for _, name in pairs]
    # 4) {"0":"healthy","1":"sick"} (map index->name)
    if isinstance(obj, dict) and all(str(k).isdigit() for k in obj.keys()):
        pairs = sorted(((int(k), str(v)) for k, v in obj.items()))
        return [name for _, name in pairs]
    raise ValueError("Unsupported JSON format for classes")

def main(json_path: str, out_path: str | None = None):
    src = Path(json_path).resolve()
    if out_path is None:
        out = src.with_name("classes.txt")
    else:
        out = Path(out_path).resolve()

    classes = load_classes_from_json(src)
    out.write_text("\n".join(classes) + "\n", encoding="utf-8")
    print(f"Wrote {len(classes)} classes -> {out}")

if __name__ == "__main__":
    # default path matches your tree:
    # training/checkpoints/apple_health/classes.json
    default_json = Path(__file__).resolve().parents[1] / "training" / "checkpoints" / "apple_health" / "classes.json"
    json_arg = sys.argv[1] if len(sys.argv) > 1 else str(default_json)
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    main(json_arg, out_arg)
