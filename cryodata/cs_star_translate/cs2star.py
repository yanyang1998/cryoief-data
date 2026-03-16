import numpy as np
from .pyem import metadata
from .pyem import star


def cs2star(input, output, minphic=None, boxsize=None, noswapxy=False, invertx=False, inverty=False):
    if isinstance(input, str):
        input = [input]

    if input[0].endswith(".cs"):
        cs = np.load(input[0])
        try:
            df = metadata.parse_cryosparc_2_cs(cs, passthroughs=input[1:], minphic=minphic,
                                               boxsize=boxsize, swapxy=noswapxy,
                                               invertx=invertx, inverty=inverty)
        except (KeyError, ValueError):
            return 1
    else:
        if len(input) > 1:
            return 1
        meta = metadata.parse_cryosparc_065_csv(input[0])
        df = metadata.cryosparc_065_csv2star(meta, minphic)

    df = star.check_defaults(df, inplace=True)
    df = star.remove_deprecated_relion2(df, inplace=True)
    star.write_star(output, df, resort_records=True, optics=True)
