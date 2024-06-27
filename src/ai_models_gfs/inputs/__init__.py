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
import os
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

        # Load the sample pressure GRIB file
        sample_pressure_grib = cml.load_source(
            "file", os.path.dirname(os.path.abspath(__file__)) + "/sample_pres.grib"
        )
        # Create a new GRIB output file for the formatted pressure data
        formatted_pressure_file = (
            f"/tmp/ai-models-gfs/gfspresformatted_{str(kwargs['date'])}_"
            f"{str(kwargs['time']).zfill(2)}.grib"
        )
        formatted_pressure_output = cml.new_grib_output(
            formatted_pressure_file, edition=1
        )
        # Construct the URL to fetch GFS pressure data
        gfs_pressure_url = (
            f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{str(kwargs['date'])}/"
            f"{str(kwargs['time']).zfill(2)}/atmos/"
            f"gfs.t{str(kwargs['time']).zfill(2)}z.pgrb2.0p25.f000"
        )
        # Load the GFS pressure data from the URL
        gfs_pressure_data = cml.load_source("url", gfs_pressure_url)

        # Iterate over the sample pressure GRIB messages
        for grib_message in sample_pressure_grib:
            parameter_name = grib_message['shortName']
            pressure_level = grib_message['level']
            template = grib_message

            # Set the date and time for the GRIB template
            eccodes.codes_set(template.handle.handle, "date", int(kwargs['date']))
            eccodes.codes_set(
                template.handle.handle, "time", int(kwargs['time']) * 100
            )

            if parameter_name == "z":
                # Select geopotential height data and convert to meters
                geopotential_height_data = gfs_pressure_data.sel(
                    param="gh", level=pressure_level
                )
                data_array = geopotential_height_data[0].to_numpy() * 9.80665
            else:
                # Select other parameters' data
                parameter_data = gfs_pressure_data.sel(
                    param=parameter_name, level=pressure_level
                )
                data_array = parameter_data[0].to_numpy()

            # Write the data to the formatted GRIB file using the template
            formatted_pressure_output.write(data_array, template=template)

        # Load the formatted GRIB file and return it
        formatted_pressure_grib = cml.load_source("file", formatted_pressure_file)
        return formatted_pressure_grib

    def sfc_load_source(self, **kwargs):

        # Load the sample surface GRIB file
        sample_surface_grib = cml.load_source(
            "file", os.path.dirname(os.path.abspath(__file__)) + "/sample_sfc.grib"
        )
        # Create a new GRIB output file for the formatted surface data
        formatted_surface_file = (
            f"/tmp/ai-models-gfs/gfssfcformatted_{str(kwargs['date'])}_"
            f"{str(kwargs['time']).zfill(2)}.grib"
        )
        formatted_surface_output = cml.new_grib_output(
            formatted_surface_file, edition=1
        )
        # Construct the URL to fetch GFS surface data
        gfs_surface_url = (
            f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{str(kwargs['date'])}/"
            f"{str(kwargs['time']).zfill(2)}/atmos/"
            f"gfs.t{str(kwargs['time']).zfill(2)}z.pgrb2.0p25.f000"
        )
        # Load the GFS surface data from the URL
        gfs_surface_data = cml.load_source("url", gfs_surface_url)

        # Iterate over the sample surface GRIB messages
        for grib_message in sample_surface_grib:
            parameter_name = grib_message['shortName']
            surface_level = grib_message['level']
            template = grib_message

            # Set the date and time for the GRIB template
            eccodes.codes_set(template.handle.handle, "date", int(kwargs['date']))
            eccodes.codes_set(
                template.handle.handle, "time", int(kwargs['time']) * 100
            )

            if parameter_name == "tp":
                # For total precipitation, create an array of zeros
                data_array = np.zeros((721, 1440))
            elif parameter_name in ["z", "lsm"]:
                # For geopotential height and land-sea mask, use the data directly
                data_array = grib_message.to_numpy()
            elif parameter_name == "msl":
                # Select mean sea level pressure data
                mean_sea_level_pressure_data = gfs_surface_data.sel(param="prmsl")
                data_array = mean_sea_level_pressure_data[0].to_numpy()
            elif parameter_name == "tcwv":
                # Select total column water vapor data
                total_column_water_vapor_data = gfs_surface_data.sel(param="pwat")
                data_array = total_column_water_vapor_data[0].to_numpy()
            else:
                # Select other parameters' data
                parameter_data = gfs_surface_data.sel(param=parameter_name)
                data_array = parameter_data[0].to_numpy()

            # Write the data to the formatted GRIB file using the template
            formatted_surface_output.write(data_array, template=template)

        # Load the formatted GRIB file and return it
        formatted_surface_grib = cml.load_source("file", formatted_surface_file)
        return formatted_surface_grib

    def ml_load_source(self, **kwargs):
        raise NotImplementedError("CDS does not support model levels")

class GdasInput(RequestBasedInput):
    WHERE = "GDAS"

    def pl_load_source(self, **kwargs):

        # Load the sample pressure GRIB file
        sample_pressure_grib = cml.load_source(
            "file", os.path.dirname(os.path.abspath(__file__)) + "/sample_pres.grib"
        )
        # Create a new GRIB output file for the formatted pressure data
        formatted_pressure_file = (
            f"/tmp/ai-models-gfs/gdaspresformatted_{str(kwargs['date'])}_"
            f"{str(kwargs['time']).zfill(2)}.grib"
        )
        if not os.path.isdir("/tmp/ai-models-gfs"):
            os.makedirs("/tmp/ai-models-gfs")
        formatted_pressure_output = cml.new_grib_output(
            formatted_pressure_file, edition=1
        )
        # Construct the URL to fetch GFS pressure data
        gdas_pressure_url = (
            f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{str(kwargs['date'])}/"
            f"{str(kwargs['time']).zfill(2)}/atmos/"
            f"gdas.t{str(kwargs['time']).zfill(2)}z.pgrb2.0p25.f000"
        )
        # Load the GFS pressure data from the URL
        gdas_pressure_data = cml.load_source("url", gdas_pressure_url)

        # Iterate over the sample pressure GRIB messages
        for grib_message in sample_pressure_grib:
            parameter_name = grib_message['shortName']
            pressure_level = grib_message['level']
            template = grib_message

            # Set the date and time for the GRIB template
            eccodes.codes_set(template.handle.handle, "date", int(kwargs['date']))
            eccodes.codes_set(
                template.handle.handle, "time", int(kwargs['time']) * 100
            )

            if parameter_name == "z":
                # Select geopotential height data and convert to meters
                geopotential_height_data = gdas_pressure_data.sel(
                    param="gh", level=pressure_level
                )
                data_array = geopotential_height_data[0].to_numpy() * 9.80665
            else:
                # Select other parameters' data
                parameter_data = gdas_pressure_data.sel(
                    param=parameter_name, level=pressure_level
                )
                data_array = parameter_data[0].to_numpy()

            # Write the data to the formatted GRIB file using the template
            formatted_pressure_output.write(data_array, template=template)

        # Load the formatted GRIB file and return it
        formatted_pressure_grib = cml.load_source("file", formatted_pressure_file)
        return formatted_pressure_grib

    def sfc_load_source(self, **kwargs):

        # Load the sample surface GRIB file
        sample_surface_grib = cml.load_source(
            "file", os.path.dirname(os.path.abspath(__file__)) + "/sample_sfc.grib"
        )
        # Create a new GRIB output file for the formatted surface data
        formatted_surface_file = (
            f"/tmp/ai-models-gfs/gdassfcformatted_{str(kwargs['date'])}_"
            f"{str(kwargs['time']).zfill(2)}.grib"
        )
        if not os.path.isdir("/tmp/ai-models-gfs"):
            os.makedirs("/tmp/ai-models-gfs")
        formatted_surface_output = cml.new_grib_output(
            formatted_surface_file, edition=1
        )
        # Construct the URL to fetch GFS surface data
        gdas_surface_url = (
            f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/gdas.{str(kwargs['date'])}/"
            f"{str(kwargs['time']).zfill(2)}/atmos/"
            f"gdas.t{str(kwargs['time']).zfill(2)}z.pgrb2.0p25.f000"
        )
        # Load the GFS surface data from the URL
        gdas_surface_data = cml.load_source("url", gdas_surface_url)

        # Iterate over the sample surface GRIB messages
        for grib_message in sample_surface_grib:
            parameter_name = grib_message['shortName']
            surface_level = grib_message['level']
            template = grib_message

            # Set the date and time for the GRIB template
            eccodes.codes_set(template.handle.handle, "date", int(kwargs['date']))
            eccodes.codes_set(
                template.handle.handle, "time", int(kwargs['time']) * 100
            )

            if parameter_name == "tp":
                # For total precipitation, create an array of zeros
                data_array = np.zeros((721, 1440))
            elif parameter_name in ["z", "lsm"]:
                # For geopotential height and land-sea mask, use the data directly
                data_array = grib_message.to_numpy()
            elif parameter_name == "msl":
                # Select mean sea level pressure data
                mean_sea_level_pressure_data = gdas_surface_data.sel(param="prmsl")
                data_array = mean_sea_level_pressure_data[0].to_numpy()
            elif parameter_name == "tcwv":
                # Select total column water vapor data
                total_column_water_vapor_data = gdas_surface_data.sel(param="pwat")
                data_array = total_column_water_vapor_data[0].to_numpy()
            else:
                # Select other parameters' data
                parameter_data = gdas_surface_data.sel(param=parameter_name)
                data_array = parameter_data[0].to_numpy()

            # Write the data to the formatted GRIB file using the template
            formatted_surface_output.write(data_array, template=template)

        # Load the formatted GRIB file and return it
        formatted_surface_grib = cml.load_source("file", formatted_surface_file)
        return formatted_surface_grib

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
