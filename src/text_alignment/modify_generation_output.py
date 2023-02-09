import sys

path_to_generation_file = sys.argv[1]
path_to_modified_generation_file = sys.argv[2]

raw_generation, correct_order = [], []
with open(path_to_generation_file, "r", encoding="utf8") as f:
    for line in f.read().splitlines():
        if line[:2] == "D-":
            correct_order.append(int(line.split(maxsplit=1)[0].split("D-")[-1]))
            splits = line.split(maxsplit=2)
            if len(splits) == 3:
                raw_generation.append(splits[2])
            else:
                raw_generation.append("")

# fix to correct order
raw_generation = [gen for _, gen in sorted(zip(correct_order, raw_generation))]

with open(path_to_modified_generation_file, "w") as f:
    for line in raw_generation:
        f.write(line + "\n")
