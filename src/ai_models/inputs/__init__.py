# (C) Copyright 2023 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from functools import cached_property

import eccodes
import climetlab as cml
import entrypoints
import numpy as np
from datetime import datetime,timedelta
from pprint import pprint
LOG = logging.getLogger(__name__)


class RequestBasedInput:
    def __init__(self, owner, **kwargs):
        self.owner = owner

    def _patch(self, **kargs):
        r = dict(**kargs)
        self.owner.patch_retrieve_request(r)
        return r

    @cached_property
    def fields_sfc(self):
        param = self.owner.param_sfc
        if not param:
            return cml.load_source("empty")

        LOG.info(f"Loading surface fields from {self.WHERE}")
        return cml.load_source(
            "multi",
            [
                self.sfc_load_source(
                    **self._patch(
                        date=date,
                        time=time,
                        param=param,
                        grid=self.owner.grid,
                        area=self.owner.area,
                        **self.owner.retrieve,
                    )
                )
                for date, time in self.owner.datetimes()
            ],
        )

    @cached_property
    def fields_pl(self):
        param, level = self.owner.param_level_pl
        if not (param and level):
            return cml.load_source("empty")

        LOG.info(f"Loading pressure fields from {self.WHERE}")
        return cml.load_source(
            "multi",
            [
                self.pl_load_source(
                    **self._patch(
                        date=date,
                        time=time,
                        param=param,
                        level=level,
                        grid=self.owner.grid,
                        area=self.owner.area,
                    )
                )
                for date, time in self.owner.datetimes()
            ],
        )

    @cached_property
    def fields_ml(self):
        param, level = self.owner.param_level_ml
        if not (param and level):
            return cml.load_source("empty")

        LOG.info(f"Loading model fields from {self.WHERE}")
        return cml.load_source(
            "multi",
            [
                self.ml_load_source(
                    **self._patch(
                        date=date,
                        time=time,
                        param=param,
                        level=level,
                        grid=self.owner.grid,
                        area=self.owner.area,
                    )
                )
                for date, time in self.owner.datetimes()
            ],
        )

    @cached_property
    def all_fields(self):
        return self.fields_sfc + self.fields_pl + self.fields_ml


class MarsInput(RequestBasedInput):
    WHERE = "MARS"

    def __init__(self, owner, **kwargs):
        self.owner = owner

    def pl_load_source(self, **kwargs):
        kwargs["levtype"] = "pl"
        logging.debug("load source mars %s", kwargs)
        return cml.load_source("mars", kwargs)

    def sfc_load_source(self, **kwargs):
        kwargs["levtype"] = "sfc"
        logging.debug("load source mars %s", kwargs)
        return cml.load_source("mars", kwargs)

    def ml_load_source(self, **kwargs):
        kwargs["levtype"] = "ml"
        logging.debug("load source mars %s", kwargs)
        return cml.load_source("mars", kwargs)

class CdsInput(RequestBasedInput):
    WHERE = "CDS"

    def pl_load_source(self, **kwargs):
        kwargs["product_type"] = "reanalysis"
        return cml.load_source("cds", "reanalysis-era5-pressure-levels", kwargs)

    def sfc_load_source(self, **kwargs):
        kwargs["product_type"] = "reanalysis"
        return cml.load_source("cds", "reanalysis-era5-single-levels", kwargs)

    def ml_load_source(self, **kwargs):
        raise NotImplementedError("CDS does not support model levels")

class GfsInput(RequestBasedInput):
    WHERE = "GFS"
    def pl_load_source(self, **kwargs):
        samplepres = cml.load_source("file","./sample_pres.grib")
        gfsformatted = cml.new_grib_output(f"./gfspresformatted_{str(kwargs['date'])}_{str(kwargs['time']).zfill(2)}.grib",edition=1)
        gfsurl = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{str(kwargs['date'])}/{str(kwargs['time']).zfill(2)}/atmos/gfs.t{str(kwargs['time']).zfill(2)}z.pgrb2.0p25.f000"
        gfspres = cml.load_source("url",f"{gfsurl}")
        for grb in samplepres:
            shortName = grb['shortName']
            level = grb['level']
            template = grb
            eccodes.codes_set(template.handle.handle, "date", int(kwargs['date']))
            eccodes.codes_set(template.handle.handle, "time", int(kwargs['time'])*100)
            if shortName=="z":
                gfsghgrbs = gfspres.sel(param="gh",level=level)
                data = gfsghgrbs[0].to_numpy()*9.80665
            else:
                gfsgrbs = gfspres.sel(param=shortName,level=level)
                data = gfsgrbs[0].to_numpy()
            gfsformatted.write(data,template=template)
        gfspresformatted = cml.load_source("file",f"./gfspresformatted_{str(kwargs['date'])}_{str(kwargs['time']).zfill(2)}.grib")            
        return gfspresformatted

    def sfc_load_source(self, **kwargs):
        samplesfc = cml.load_source("file","./sample_sfc.grib")
        gfsformatted = cml.new_grib_output(f"./gfssfcformatted_{str(kwargs['date'])}_{str(kwargs['time']).zfill(2)}.grib",edition=1)
        gfsurl = f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{str(kwargs['date'])}/{str(kwargs['time']).zfill(2)}/atmos/gfs.t{str(kwargs['time']).zfill(2)}z.pgrb2.0p25.f000"
        gfssfc = cml.load_source("url",f"{gfsurl}")
        for grb in samplesfc:
            shortName = grb['shortName']
            level = grb['level']
            template = grb
            eccodes.codes_set(template.handle.handle, "date", int(kwargs['date']))
            eccodes.codes_set(template.handle.handle, "time", int(kwargs['time'])*100)
            if shortName=="tp":
                data = np.zeros((721,1440))
            elif shortName=="z" or shortName=="lsm":
                data = grb.to_numpy()
            elif shortName=="msl":
                gfsmslgrbs = gfssfc.sel(param="prmsl")
                data = gfsmslgrbs[0].to_numpy()
            elif shortName=="tcwv":
                gfstcwvgrbs = gfssfc.sel(param="pwat")
                data = gfstcwvgrbs[0].to_numpy()
            else:
                gfsgrbs = gfssfc.sel(param=shortName)
                data = gfsgrbs[0].to_numpy()
            gfsformatted.write(data,template=template)
        gfssfcformatted = cml.load_source("file",f"./gfssfcformatted_{str(kwargs['date'])}_{str(kwargs['time']).zfill(2)}.grib")
        return gfssfcformatted

    def ml_load_source(self, **kwargs):
        raise NotImplementedError("CDS does not support model levels")

class OpenDataInput(RequestBasedInput):
    WHERE = "OPENDATA"

    RESOLS = {(0.25, 0.25): "0p25"}

    def __init__(self, owner, **kwargs):
        self.owner = owner

    def _adjust(self, kwargs):
        if "level" in kwargs:
            # OpenData uses levelist instead of level
            kwargs["levelist"] = kwargs.pop("level")

        grid = kwargs.pop("grid")
        if isinstance(grid, list):
            grid = tuple(grid)

        kwargs["resol"] = self.RESOLS[grid]
        r = dict(**kwargs)
        r.update(self.owner.retrieve)
        return r

    def pl_load_source(self, **kwargs):
        self._adjust(kwargs)
        kwargs["levtype"] = "pl"
        logging.debug("load source ecmwf-open-data %s", kwargs)
        return cml.load_source("ecmwf-open-data", **kwargs)

    def sfc_load_source(self, **kwargs):
        self._adjust(kwargs)
        kwargs["levtype"] = "sfc"
        logging.debug("load source ecmwf-open-data %s", kwargs)
        return cml.load_source("ecmwf-open-data", **kwargs)

    def ml_load_source(self, **kwargs):
        self._adjust(kwargs)
        kwargs["levtype"] = "ml"
        logging.debug("load source ecmwf-open-data %s", kwargs)
        return cml.load_source("ecmwf-open-data", **kwargs)


class FileInput:
    def __init__(self, owner, file, **kwargs):
        self.file = file
        self.owner = owner

    @cached_property
    def fields_sfc(self):
        return self.all_fields.sel(levtype="sfc")

    @cached_property
    def fields_pl(self):
        return self.all_fields.sel(levtype="pl")

    @cached_property
    def fields_ml(self):
        return self.all_fields.sel(levtype="ml")

    @cached_property
    def all_fields(self):
        return cml.load_source("file", self.file)


def get_input(name, *args, **kwargs):
    return available_inputs()[name].load()(*args, **kwargs)


def available_inputs():
    result = {}
    for e in entrypoints.get_group_all("ai_models.input"):
        result[e.name] = e
    return result
