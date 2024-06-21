import climetlab as cml
import eccodes
import numpy as np
output = cml.new_grib_output(f"./tp.grib")
tp = cml.load_source("file","./adaptor.mars.internal-1718950358.0217638-32715-18-d2504d39-d0a9-40ef-a9ae-9660a9d0c312.grib")
tpzero = np.zeros((721, 1440))
for grb in tp:
    template = grb
    eccodes.codes_set(template.handle.handle, "shortName", "tp")
output.write(tpzero,template=template)
