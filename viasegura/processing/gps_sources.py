import datetime as dt
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pynmea2
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from scipy.interpolate import interp1d
from shapely.geometry import LineString
from tqdm import tqdm

from viasegura.analyzers.utils import total_distance
from viasegura.processing.exceptions import InvalidGPSData
from viasegura.processing.utils import decimal_coords

logger = logging.getLogger(__name__)


class GPS_Processer:
    def __init__(self):
        self._calculate_seconds_from_start()

    def _calculate_seconds_from_start(self):
        self.gps_df["seconds_from_start"] = self.gps_df.seconds.values - self.gps_df.seconds.values[0]

    def adjust_gps_data(self, number_images):
        list_values = np.linspace(0, self.gps_df.seconds_from_start.max(), number_images).astype("float")
        f2 = interp1d(self.gps_df["seconds_from_start"].values, self.gps_df["longitude"].values, kind="linear")
        f3 = interp1d(self.gps_df["seconds_from_start"].values, self.gps_df["latitude"].values, kind="linear")
        initial_latitud = self.gps_df.loc[0].latitude
        initial_longitude = self.gps_df.loc[0].longitude
        final_longitude = f2(list_values)
        final_latitude = f3(list_values)
        final_longitude[0] = initial_longitude
        final_latitude[0] = initial_latitud
        self.gps_df = pd.DataFrame({"latitude": final_latitude, "longitude": final_longitude})

    def generate_gps_metrics(self, min_distance_group):
        latitudes = list(self.gps_df.latitude.values)
        longitudes = list(self.gps_df.longitude.values)
        distances = total_distance(latitudes[:-1], longitudes[:-1], latitudes[1:], longitudes[1:])
        self.distances = [0]
        self.distances += list(distances)
        distances = np.append(distances, distances[-1])

        id_distances = []
        id_dist = 0
        sum_dist = 0
        self.section_latitude = [latitudes[0]]
        self.section_longitude = [longitudes[0]]
        self.section_distances = []
        for i in range(len(distances)):
            if sum_dist > min_distance_group:
                id_dist += 1
                self.section_distances.append(sum_dist)
                sum_dist = 0
                self.section_latitude.append(latitudes[i])
                self.section_longitude.append(longitudes[i])
            sum_dist += distances[i]
            id_distances.append(id_dist)
        self.section_distances.append(sum_dist)
        self.section_latitude.append(latitudes[i])
        self.section_longitude.append(longitudes[i])
        self.gps_df["distances"] = distances
        self.gps_df["section"] = id_distances
        self.cumulative_sum = np.cumsum(self.distances)
        self.gps_df["cumulative_sum"] = self.cumulative_sum
        max_value = int(np.max(self.cumulative_sum) // 20) * 20
        self.image_numbers = []
        for i in list(range(0, max_value, 20)):
            self.image_numbers.append(np.argmin(abs(self.cumulative_sum - i)))
        self.usable_gps_df = self.gps_df.loc[self.image_numbers]
        self.usable_gps_df["idx_values"] = self.usable_gps_df.index
        self.usable_gps_df = self.usable_gps_df.reset_index(drop=True)
        self.usable_gps_df["section_id"] = [i // 5 for i in range(self.usable_gps_df.shape[0])]
        self.gps_df["geometry"] = list(map(lambda long, lat: [long, lat], self.gps_df.longitude.values, self.gps_df.latitude.values))
        self.gps_group_info = self.usable_gps_df.groupby(["section_id"]).aggregate({"idx_values": "min"}).reset_index()
        self.gps_df["idx_ind"] = self.gps_df.index
        self.gps_df["section_id"] = np.digitize(self.gps_df.idx_ind.values, self.gps_group_info.idx_values.values) - 1
        self.gps_group_info = pd.merge(
            self.gps_group_info,
            self.gps_df.groupby(["section_id"]).aggregate({"geometry": list, "distances": "sum"}).reset_index(),
            on="section_id",
            how="left",
        )
        self.gps_group_info["geometry"] = list(map(lambda x: LineString(x), self.gps_group_info.geometry))
        self.gps_group_info = self.gps_group_info.drop(["idx_values"], axis=1)


class GPS_Standard_Loader(GPS_Processer):
    def __init__(self, route, **kwargs):
        self.route = Path(route)
        self.gps_df = None
        self.load_gps_data()
        super().__init__()

    def load_gps_data(self):
        data = []
        with open(self.route) as file:
            for line in file.readlines():
                try:
                    data.append(pynmea2.parse(line))
                except pynmea2.ParseError:
                    continue
        gps = {}
        for value in data:
            salida = gps.get(value.sentence_type, None)
            if salida is None:
                gps[value.sentence_type] = []
            gps[value.sentence_type].append(value)
        select = list(gps.keys())[0]
        gps_list = gps[select]
        self.gps_df = pd.DataFrame(
            list(map(lambda x: (x.timestamp, x.longitude, x.latitude), gps_list)), columns=["timestamp", "longitude", "latitude"]
        )
        self.gps_df = self.gps_df.drop_duplicates(subset=["longitude", "latitude"]).reset_index(drop=True)
        if isinstance(self.gps_df.loc[0].timestamp, str):
            self.gps_df["seconds"] = list(map(lambda x: int(float(x)) / 1000, self.gps_df.timestamp.values))
        else:
            self.gps_df["seconds"] = list(map(lambda x: (x.hour * 3600) + (x.minute * 60) + (x.second), self.gps_df.timestamp.values))
        self.gps_df = self.gps_df.groupby(["seconds", "timestamp"]).aggregate({"latitude": "first", "longitude": "first"}).reset_index()


class GPS_CSV_Loader(GPS_Processer):
    def __init__(self, route, **kwargs):
        latitud_column = kwargs.get("latitud_column", None)
        longitud_column = kwargs.get("longitud_column", None)
        time_column = kwargs.get("time_column", None)
        date_column = kwargs.get("date_column", None)
        decimal_character = kwargs.get("decimal_character", ",")

        self.route = Path(route)
        self.columns_names = []
        if time_column is not None:
            self.columns_names.append(time_column)
        if date_column is not None:
            self.columns_names.append(date_column)
        if longitud_column is not None:
            self.columns_names.append(longitud_column)
        if latitud_column is not None:
            self.columns_names.append(latitud_column)
        self.decimal_character = decimal_character
        self.load_gps_data()
        super().__init__()

    def load_gps_data(self):
        self.gps_df = pd.read_csv(self.route, sep=";", encoding="latin1", decimal=self.decimal_character)[self.columns_names]
        if self.columns_names == 3:
            self.gps_df.columns = ["timestamp", "longitude", "latitude"]
            self.gps_df["timestamp"] = list(map(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), self.gps_df["timestamp"]))
        else:
            self.gps_df.columns = ["time", "date", "longitude", "latitude"]
            self.gps_df["timestamp"] = list(
                map(
                    lambda x, y: dt.datetime.strptime(x + " " + y, "%Y-%m-%d %H:%M:%S"),
                    self.gps_df["date"].values,
                    self.gps_df["time"].values,
                )
            )
            self.gps_df = self.gps_df[["timestamp", "longitude", "latitude"]]
        if isinstance(self.gps_df.loc[0].timestamp, str):
            self.gps_df["seconds"] = list(map(lambda x: int(float(x)) / 1000, self.gps_df.timestamp.values))
        else:
            self.gps_df["seconds"] = list(map(lambda x: (x.hour * 3600) + (x.minute * 60) + (x.second), self.gps_df.timestamp))


class GPS_Image_Route_Loader(GPS_Processer):
    def __init__(self, images_routes, **kwargs):
        self.routes = images_routes
        self.load_gps_data()
        super().__init__()

    def load_gps_data(self):
        self.gps_df = pd.DataFrame(list(tqdm(map(lambda img_path: self.load_single_value(img_path), self.routes))))
        self.gps_df["seconds"] = list(map(lambda x: (x.hour * 3600) + (x.minute * 60) + (x.second), self.gps_df.timestamp))

    def load_exif_data(self, img_path):
        image = Image.open(img_path)
        exifdata = image.getexif()

        d = {}
        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            try:
                if isinstance(data, bytes):
                    data = data.decode()
                d[tag] = data
            except ValueError:  # noqa: E722
                pass
        return d

    def get_gpsdata_from_exif(self, filename):
        exif = Image.open(filename)._getexif()
        exif_data = {}
        gps_data = {}
        if exif is not None:
            for key, value in exif.items():
                name = TAGS.get(key, key)
                try:
                    data = exif[key].decode() if isinstance(exif[key], bytes) else exif[key]
                    exif_data[name] = data
                except:  # noqa: E722
                    logger.debug(f"Error decoding exif in key: {key}")

            if "GPSInfo" in exif_data:
                for key in exif_data["GPSInfo"].keys():
                    name = GPSTAGS.get(key, key)
                    gps_data[name] = exif_data["GPSInfo"].get(key)

            gps_data["DateTimeOriginal"] = exif_data.get("DateTimeOriginal", None)

        return gps_data

    def load_single_value(self, img_path):
        try:
            # get from exif
            d = self.load_exif_data(str(img_path))
            lat = np.array(d["GPSInfo"][2])
            lon = np.array(d["GPSInfo"][4])
            lat = sum(np.array(lat[:, 0] / lat[:, 1]) * np.array([1.0, 1.0 / 60.0, 1.0 / 3600.0])) * (-1 if d["GPSInfo"][1] == "S" else 1)
            lon = sum(np.array(lon[:, 0] / lon[:, 1]) * np.array([1.0, 1.0 / 60.0, 1.0 / 3600.0])) * (-1 if d["GPSInfo"][3] == "W" else 1)
            time = dt.datetime.strptime(d["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
        except:  # noqa: E722
            try:
                # get from gpsinfo
                d = self.get_gpsdata_from_exif(str(img_path))
                lat = decimal_coords(d["GPSLatitude"], d["GPSLatitudeRef"])
                lon = decimal_coords(d["GPSLongitude"], d["GPSLongitudeRef"])
                time = dt.datetime.strptime(d["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
            except:  # noqa: E722
                raise InvalidGPSData(f"Could not load GPS data from image: {img_path}")

        return {"timestamp": time, "longitude": lon, "latitude": lat}

    # def load_single_value(self, img_path):
    #     d = {}
    #     image = Image.open(img_path)
    #     exifdata = image.getexif()
    #     for tag_id in exifdata:
    #         tag = TAGS.get(tag_id, tag_id)
    #         data = exifdata.get(tag_id)
    #         try:
    #             if isinstance(data, bytes):
    #                 data = data.decode()
    #             d[tag] = data
    #         except:
    #             pass
    #     lat = np.array(d["GPSInfo"][2])
    #     lon = np.array(d["GPSInfo"][4])
    #     lat = sum(np.array(lat[:, 0] / lat[:, 1]) * np.array([1.0, 1.0 / 60.0, 1.0 / 3600.0])) * (-1 if d["GPSInfo"][1] == "S" else 1)
    #     lon = sum(np.array(lon[:, 0] / lon[:, 1]) * np.array([1.0, 1.0 / 60.0, 1.0 / 3600.0])) * (-1 if d["GPSInfo"][3] == "W" else 1)
    #     time = dt.datetime.strptime(d["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
    #     return {"timestamp": time, "longitude": lon, "latitude": lat}


class GPS_Image_Folder_Loader(GPS_Image_Route_Loader):
    def __init__(self, route, **kwargs):
        self.routes = [Path(route) / item for item in os.listdir(Path(route))]
        self.load_gps_data()
        super(GPS_Image_Route_Loader, self).__init__()


gps_source_options_dict = {
    "image_routes": GPS_Image_Route_Loader,
    "image_folder": GPS_Image_Folder_Loader,
    "csv": GPS_CSV_Loader,
    "loc": GPS_Standard_Loader,
}


def GPS_Data_Loader(source_type, gps_in, **kwargs):
    if source_type not in gps_source_options_dict:
        raise NameError(f"{source_type} not implemented on the method")
    return gps_source_options_dict[source_type](gps_in, **kwargs)
