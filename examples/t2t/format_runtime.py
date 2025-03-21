"""Script for formatting runtime numbers in a LaTeX and markdown table.

Example usage:
```bash
python format_runtime.py -op PATH_TO_OUTPUT_DIR
```
"""

import torch
import os
import pandas as pd
from tabulate import tabulate

from util_experiments import get_base_parser

parser = get_base_parser()
parser.add_argument("--batch_size", "-bs", default=64, type=int, help="batch size")
parser.add_argument(
    "--batch_number", "-bn", default=1, type=int, help="batch number >= 1"
)
parser.add_argument(
    "--num_runs",
    "-n",
    default=1,
    type=int,
    help="number of runs (excluding warm-up runs)",
)
args, opt = parser.parse_known_args()
device = args.device if args.device else torch.device
batch_size = args.batch_size
batch_number = args.batch_number
output_path = args.output_path
num_runs = args.num_runs

# %%
# load files from output/times/
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
    df = pd.DataFrame()
    # iterate over csv files including method name
    # for save_path in glob.glob(os.path.join(output_path, "times", f"times-n{num_runs}-{method}-{device}-bs{batch_size}-bn*.csv")):
    for bn in range(1, 50 + 1):
        save_path = os.path.join(
            output_path,
            "times",
            f"times-n{num_runs}-{method}-{device}-bs{batch_size}-bn{bn}.csv",
        )
        print(f"Reading from {save_path}...")
        # read csv file using pandas
        df = pd.concat(
            [df, pd.read_csv(save_path, index_col=0)], axis=0
        )  # note: times are in milliseconds

    # print(df)
    # get mean and std
    mean_data[method] = df.mean()
    std_data[method] = df.std()

# %%
mean_df = pd.DataFrame(data=mean_data)
std_df = pd.DataFrame(data=std_data)
# create new dataframe where each entry has the format f"{mean:.2f} ± {std:.2f}"
mean_std_df = (
    mean_df.apply(lambda x: x.map("{:.2f}".format))
    + " ± "
    + std_df.apply(lambda x: x.map("{:.2f}".format))
)
# only select the rows with index 'attention1.attn.attention_layer', 'attention2.attn.attention_layer'
mean_std_df = mean_std_df.loc[
    ["attention1.attn.attention_layer", "attention2.attn.attention_layer"]
]

# %%
# convert mean_std_df to latex using pandas
print("Absolute times:")
print(mean_std_df.to_latex())
print(tabulate(mean_std_df, tablefmt="github", headers="keys"))
