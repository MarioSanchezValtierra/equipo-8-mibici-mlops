"""Fuentes externas con caché: clima por zona (Open-Meteo) y lugares de interés (OpenStreetMap)."""

import json
import logging
from pathlib import Path

import osmium
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import transform

from mibici_mlops.config import (
    DATA_END,
    DATA_START,
    METRIC_CRS,
    OSM_PATH,
    OSM_URL,
    POI_BBOX_MARGIN,
    POI_CATEGORIES,
    POI_DIR,
    POI_PATH,
    POI_RADIUS_METERS,
    STATION_ZONES_PATH,
    WEATHER_COLUMNS,
    WEATHER_DIR,
    WEATHER_GRID_DEGREES,
    WEATHER_PATH,
    WEATHER_TIMEZONE,
    WEATHER_URL,
)

logger = logging.getLogger("mibici_mlops")


def assign_weather_zones(stations: pd.DataFrame, grid_degrees: float) -> pd.DataFrame:
    # Asignar cada estación a una celda de la rejilla.
    zones = stations.drop_duplicates(subset="Origen_Id").copy()
    zones["latitude_zone"] = (zones["latitude"] / grid_degrees).round() * grid_degrees
    zones["longitude_zone"] = (zones["longitude"] / grid_degrees).round() * grid_degrees
    zones["weather_zone"] = zones.groupby(["latitude_zone", "longitude_zone"]).ngroup() + 1
    return zones


def weather_zone_points(zones: pd.DataFrame) -> pd.DataFrame:
    # Coordenadas centrales de cada zona.
    return zones.groupby("weather_zone", as_index=False).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        station_count=("Origen_Id", "count"),
    )


def download_zone_weather(points, start, end, url, variables, timezone_name):
    # Descargar el clima diario de cada zona y validar el resultado.
    records = []
    for zone in points.itertuples(index=False):
        params = {
            "latitude": float(zone.latitude),
            "longitude": float(zone.longitude),
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "daily": ",".join(variables),
            "timezone": timezone_name,
        }
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        zone_weather = pd.DataFrame(response.json()["daily"])
        zone_weather["weather_zone"] = zone.weather_zone
        zone_weather["latitude"] = zone.latitude
        zone_weather["longitude"] = zone.longitude
        records.append(zone_weather)
        logger.info("Zona %s: %s días descargados", zone.weather_zone, len(zone_weather))

    weather = pd.concat(records, ignore_index=True)
    weather = weather[["time", "weather_zone", "latitude", "longitude"] + variables]
    expected_rows = len(pd.date_range(start, end)) * len(points)
    if len(weather) != expected_rows:
        raise ValueError("La cantidad de registros meteorológicos no coincide.")
    if weather.isna().any().any():
        raise ValueError("Se encontraron valores meteorológicos faltantes.")
    if weather.duplicated(["time", "weather_zone"]).any():
        raise ValueError("Se encontraron fechas duplicadas por zona.")
    return weather


def load_or_download_weather(stations: pd.DataFrame, force: bool = False):
    # Leer el clima guardado o descargarlo si todavía no existe.
    if WEATHER_PATH.is_file() and STATION_ZONES_PATH.is_file() and not force:
        weather = pd.read_csv(WEATHER_PATH, parse_dates=["time"])
        zones = pd.read_csv(STATION_ZONES_PATH)
        expected_ids = set(stations["Origen_Id"])
        if (
            expected_ids.issubset(set(zones["Origen_Id"]))
            and not zones["Origen_Id"].duplicated().any()
            and not weather.duplicated(["time", "weather_zone"]).any()
            and set(WEATHER_COLUMNS).issubset(weather.columns)
            and weather["time"].min() <= DATA_START
            and weather["time"].max() >= DATA_END
        ):
            logger.info("Clima reutilizado: %s", WEATHER_PATH.name)
            return weather, zones
        logger.warning("El caché meteorológico está incompleto; se reconstruye.")

    zones = assign_weather_zones(stations, WEATHER_GRID_DEGREES)
    weather = download_zone_weather(
        weather_zone_points(zones), DATA_START, DATA_END,
        WEATHER_URL, WEATHER_COLUMNS, WEATHER_TIMEZONE,
    )
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    weather.to_csv(WEATHER_PATH, index=False)
    zones[["Origen_Id", "weather_zone"]].to_csv(STATION_ZONES_PATH, index=False)
    weather["time"] = pd.to_datetime(weather["time"])
    return weather, zones[["Origen_Id", "weather_zone"]]


def matching_categories(tags: dict, categories: dict) -> list[str]:
    # Categorías cuyas etiquetas coinciden con las del elemento.
    matched = []
    for category, filters in categories.items():
        for key, values in filters.items():
            if key in tags and (values is None or tags[key] in values):
                matched.append(category)
                break
    return matched


def inside_bbox(longitude: float, latitude: float, bbox: list[float]) -> bool:
    return bbox[0] <= longitude <= bbox[2] and bbox[1] <= latitude <= bbox[3]


class POINodeExtractor(osmium.SimpleHandler):
    # Lugares de interés representados como puntos.
    def __init__(self, bbox, categories):
        super().__init__()
        self.bbox = bbox
        self.categories = categories
        self.records = []

    def node(self, node):
        if not node.location.valid():
            return
        longitude, latitude = node.location.lon, node.location.lat
        if not inside_bbox(longitude, latitude, self.bbox):
            return
        tags = dict(node.tags)
        for category in matching_categories(tags, self.categories):
            self.records.append({
                "osm_type": "node", "osm_id": node.id, "category": category,
                "name": tags.get("name"), "latitude": latitude, "longitude": longitude,
                "geometry": Point(longitude, latitude),
            })


class POIAreaExtractor(osmium.SimpleHandler):
    # Lugares de interés representados como polígonos.
    def __init__(self, bbox, categories):
        super().__init__()
        self.bbox = bbox
        self.categories = categories
        self.records = []
        self.factory = osmium.geom.GeoJSONFactory()

    def area(self, area):
        tags = dict(area.tags)
        matched = matching_categories(tags, self.categories)
        if not matched:
            return
        try:
            polygon = shape(json.loads(self.factory.create_multipolygon(area)))
        except (RuntimeError, ValueError):
            return
        if polygon.is_empty:
            return
        center = polygon.representative_point()
        if not inside_bbox(center.x, center.y, self.bbox):
            return
        for category in matched:
            self.records.append({
                "osm_type": "area", "osm_id": area.orig_id(), "category": category,
                "name": tags.get("name"), "latitude": center.y, "longitude": center.x,
                "geometry": polygon,
            })


def download_osm_extract(url: str, path: Path) -> Path:
    # Descargar el extracto de OpenStreetMap solo si no existe.
    if path.is_file() and path.stat().st_size > 0:
        logger.info("Extracto OSM reutilizado: %s", path.name)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)
    logger.info("Extracto OSM descargado: %s", path)
    return path


def extract_pois(osm_path: Path, bbox: list[float], categories: dict) -> pd.DataFrame:
    # Leer nodos y áreas del extracto y quitar duplicados.
    nodes = POINodeExtractor(bbox, categories)
    nodes.apply_file(str(osm_path))
    areas = POIAreaExtractor(bbox, categories)
    areas.apply_file(str(osm_path), locations=True, idx="flex_mem")
    pois = pd.DataFrame(nodes.records + areas.records)
    return pois.drop_duplicates(subset=["osm_type", "osm_id", "category"])


def count_pois_near_stations(stations, pois, categories, radius, metric_crs) -> pd.DataFrame:
    # Contar por estación y categoría los POI dentro del radio.
    transformer = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    projected = [transform(transformer.transform, geometry) for geometry in pois["geometry"]]
    poi_categories = pois["category"].to_numpy()
    rows = []
    for station in stations.itertuples(index=False):
        station_point = Point(*transformer.transform(station.longitude, station.latitude))
        nearby = [
            category for category, geometry in zip(poi_categories, projected)
            if station_point.distance(geometry) <= radius
        ]
        counts = pd.Series(nearby, dtype="object").value_counts()
        rows.append({
            "Origen_Id": station.Origen_Id,
            **{f"{category}_count": int(counts.get(category, 0)) for category in categories},
        })
    return pd.DataFrame(rows)


def load_or_build_poi_counts(stations: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    # Leer los conteos guardados o calcularlos desde OpenStreetMap.
    if POI_PATH.is_file() and not force:
        logger.info("POI reutilizados: %s", POI_PATH.name)
        return pd.read_csv(POI_PATH)

    stations = stations.drop_duplicates(subset="Origen_Id")
    bbox = [
        stations["longitude"].min() - POI_BBOX_MARGIN,
        stations["latitude"].min() - POI_BBOX_MARGIN,
        stations["longitude"].max() + POI_BBOX_MARGIN,
        stations["latitude"].max() + POI_BBOX_MARGIN,
    ]
    download_osm_extract(OSM_URL, OSM_PATH)
    pois = extract_pois(OSM_PATH, bbox, POI_CATEGORIES)
    counts = count_pois_near_stations(
        stations, pois, POI_CATEGORIES, POI_RADIUS_METERS, METRIC_CRS
    )
    POI_DIR.mkdir(parents=True, exist_ok=True)
    counts.to_csv(POI_PATH, index=False)
    return counts
