from __future__ import annotations

from pathlib import Path
import sys
import yaml

# 允许从 examples/ 目录直接运行
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nemoasr2pytorch.text.tokenizer import TextTokenizer


def main() -> None:
    cfg_path = REPO_ROOT / "parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2/model_config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    labels = cfg["labels"]
    joint_num_classes = cfg["joint"]["num_classes"]

    model_dir = REPO_ROOT / "parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2"
    tokenizer = TextTokenizer.from_nemo_config_dir(model_dir)

    print("Tokenizer vocab_size:", tokenizer.vocab_size)
    print("Config labels_len   :", len(labels))
    print("Joint.num_classes   :", joint_num_classes)

    text = "hello world"
    ids = tokenizer.encode(text)
    rec = tokenizer.decode(ids)
    print("Example encode/decode:")
    print("  text   :", text)
    print("  ids    :", ids[:20], "...")
    print("  decoded:", rec)


if __name__ == "__main__":
    main()

