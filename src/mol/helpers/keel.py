import io
import re

import pandas as pd


def parse_keel_dat(dat_file):
    with open(dat_file, "r") as fp:
        data = fp.read()
        header, payload = data.split("@data\n")

    relation = re.findall(r"@[Rr]elation (.*)", header)[0]
    attributes = re.findall(r"@[Aa]ttribute (.*?)[ {](integer|real|.*)", header)
    output = re.findall(r"@[Oo]utput[s]? (.*)", header)

    dtype_map = {"integer": int, "real": float}

    columns, types = zip(*attributes)
    types = [*map(lambda _: dtype_map.get(_, "category"), types)]
    dtype = dict(zip(columns, types))

    data = pd.read_csv(
        io.StringIO(payload), names=columns, dtype=dtype, na_values=["<null>", "?"]
    )

    if not output:  # if it was not found
        output = columns[-1]

    target = data[output].iloc[:, 0].str.strip()
    data.drop(labels=output, axis=1, inplace=True)

    return relation, data, target
