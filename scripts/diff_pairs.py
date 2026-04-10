#!/usr/bin/env python3
import re
import sys

def extract_pairs(filename):
    pairs = set()
    with open(filename) as f:
        for line in f:
            m = re.search(r'\((\d+)\s+(\d+)\)', line)
            if m:
                pairs.add((int(m.group(1)), int(m.group(2))))
    return pairs

if len(sys.argv) == 3:
    file_a, file_b = sys.argv[1], sys.argv[2]
else:
    file_a, file_b = 'orig', 'finetune'

orig = extract_pairs(file_a)
finetune = extract_pairs(file_b)

only_orig = sorted(orig - finetune)
only_finetune = sorted(finetune - orig)
both = sorted(orig & finetune)

print(f"Total in {file_a}:     {len(orig)}")
print(f"Total in {file_b}: {len(finetune)}")
print(f"In both:           {len(both)}")
print(f"Only in orig:      {len(only_orig)}")
print(f"Only in finetune:  {len(only_finetune)}")

if only_orig:
    print(f"\nPairs only in {file_a}:")
    for p in only_orig:
        print(f"  {p}")

if only_finetune:
    print(f"\nPairs only in {file_b}:")
    for p in only_finetune:
        print(f"  {p}")

if both:
    print("\nPairs in both:")
    for p in both:
        print(f"  {p}")
