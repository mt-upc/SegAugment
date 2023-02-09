import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-dir", "-ckpt", required=True, type=str)
parser.add_argument("--num-ckpts", "-n", default=10, type=int)
parser.add_argument("--minimize-metric", "-min", action="store_true")
args = parser.parse_args()

ckpt_dir = Path(args.ckpt_dir)

sorted_ckpt_names = sorted(ckpt_dir.glob("checkpoint.best*"))
if not args.minimize_metric:
    sorted_ckpt_names = sorted_ckpt_names[::-1]

with open(ckpt_dir / f"best_{args.num_ckpts}.txt", "w") as f:
    for i, ckpt_name in enumerate(sorted_ckpt_names[: args.num_ckpts]):
        f.write(str(ckpt_name))
        if i != args.num_ckpts - 1:
            f.write(" ")
