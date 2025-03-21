"""Script for formatting accuracy numbers in a LaTeX and markdown table.

Example usage:
```bash
python format_accuracy.py -op PATH_TO_OUTPUT_DIR
```
"""

import torch
import os
import pandas as pd
import glob
from tabulate import tabulate

from util_experiments import get_base_parser

parser = get_base_parser()
args, opt = parser.parse_known_args()
device = args.device if args.device else torch.device
output_path = args.output_path
seed = args.seed

# load files from OUTPUT_PATH/acc/
methods = [
    "full",
    "performer",
    "reformer",
    "scatterbrain",
    "kdeformer",
    "thinformer",
]

mean_data = {}
std_data = {}
for method in methods:
    seeds = []
    # iterate over csv files including method name
    for save_path in glob.glob(
        os.path.join(output_path, "acc", f"acc-{method}-{method}-{device}-s*.csv")
    ):
        print(f"Reading {save_path}")
        df = pd.read_csv(save_path, index_col=0)
        # compute column means
        df_mean = df.mean()
        # add enty for accuracy
        df_mean["accuracy"] = df["corrects"].sum() / df["cnt"].sum() * 100
        # concat to df_all
        seeds.append(df_mean)
    if len(seeds) == 0:
        print(f"No accuracy data found for {method}")
        continue
    # concat all seeds
    df_all = pd.concat(seeds, axis=1).T
    # compute mean and std across all batches and seeds
    mean_data[method] = df_all.mean()
    std_data[method] = df_all.std()

mean_df = pd.DataFrame(data=mean_data)
std_df = pd.DataFrame(data=std_data)
# create new dataframe where each entry has the format f"{mean:.2f} ± {std:.2f}"
mean_std_df = (
    mean_df.apply(lambda x: x.map("{:.2f}".format))
    + " ± "
    + std_df.apply(lambda x: x.map("{:.2f}".format))
)

# convert mean_std_df to latex using pandas
print("Accuracies:")
print(mean_std_df.transpose().to_latex())
print(tabulate(mean_std_df.transpose(), tablefmt="github", headers="keys"))
