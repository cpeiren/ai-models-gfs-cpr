import climetlab as cml
#data = cml.load_source("cds", "reanalysis-era5-pressure-levels",{'date': 20230605, 'time': 0, 'param': ['t', 'z', 'u', 'v', 'w', 'q'], 'level': [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], 'grid': [0.25, 0.25], 'area': [90, 0, -90, 360], 'type': 'fc', 'stream': 'oper', 'product_type': 'reanalysis'})
#print(data.to_xarray())
data = cml.load_source("url", "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.20240620/12/atmos/gfs.t12z.pgrb2.0p25.f000")
datasub = data.sel(param="prmsl")
for thing in datasub:
    print(thing)
