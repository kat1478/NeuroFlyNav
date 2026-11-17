"""Build the "fly in the city" map of central Warsaw from OpenStreetMap.

Run in the `neuroflynav-geo` environment (osmnx, geopandas, tippecanoe from conda-forge):

    conda run -n neuroflynav-geo python geo/build_map.py

Source: the dated Geofabrik extract of the Mazowieckie region (downloaded once into the
gitignored data/osm/, MD5 checked), clipped to a fixed bounding box with osmium and read with
osmnx. (The public Overpass API asks automated tools not to use it, so we do not.)
Writes:
- app/public/map/warsaw.pmtiles  vector tiles: buildings, roads, parks, water (for MapLibre);
- app/public/data/treats.geojson treat points: bakeries, cafés, ice cream, confectioneries;
- app/public/map/manifest.json   bounding box, retrieval time, versions, SHA-256 of outputs.

Map data © OpenStreetMap contributors, available under the Open Database License (ODbL).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import osmnx as ox
from shapely.geometry import box

# Central Warsaw (Śródmieście and Powiśle, with a stretch of the Vistula): ~2.7 km × 2.5 km.
BBOX = (21.000, 52.222, 21.040, 52.245)  # west, south, east, north (WGS84)

LAYERS: dict[str, dict[str, object]] = {
    "water": {"natural": "water", "waterway": ["riverbank", "river"]},
    "parks": {"leisure": ["park", "garden"], "landuse": ["grass", "forest"]},
    "roads": {"highway": True},
    "buildings": {"building": True},
}
TREATS: dict[str, object] = {
    "amenity": ["cafe", "ice_cream"],
    "shop": ["bakery", "confectionery", "pastry"],
}
KEEP = {
    "water": ["natural", "waterway"],
    "parks": ["leisure", "landuse", "name"],
    "roads": ["highway"],
    "buildings": [],
}
EXTRACT = (
    "mazowieckie-261116.osm.pbf"  # dated, immutable Geofabrik file (2026-11-16 data)
)
EXTRACT_URL = f"https://download.geofabrik.de/europe/poland/{EXTRACT}"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "osm"
MAP_DIR = ROOT / "app" / "public" / "map"
DATA_DIR = ROOT / "app" / "public" / "data"
MAX_PMTILES_BYTES = 10 * 1024 * 1024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()  # Geofabrik publishes MD5 checksums
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_osm() -> Path:
    """Download the regional extract once, check its MD5, clip it to BBOX as OSM XML."""
    pbf = RAW / EXTRACT
    if not pbf.exists():
        print(f"downloading {EXTRACT_URL}", file=sys.stderr)
        urllib.request.urlretrieve(EXTRACT_URL, pbf)  # fixed https URL
        with urllib.request.urlopen(f"{EXTRACT_URL}.md5") as response:
            expected = response.read().decode().split()[0]
        if md5(pbf) != expected:
            pbf.unlink()
            raise SystemExit(f"MD5 mismatch for {EXTRACT}")
    clipped = RAW / "warsaw-centre.osm"
    if not clipped.exists():
        west, south, east, north = BBOX
        subprocess.run(
            [
                "osmium",
                "extract",
                f"--bbox={west},{south},{east},{north}",
                "--strategy=smart",
                "--overwrite",
                f"--output={clipped}",
                str(pbf),
            ],
            check=True,
        )
    return clipped


def fetch(osm: Path, tags: dict[str, object]) -> gpd.GeoDataFrame:
    """OSM features with the given tags, clipped to the bounding box."""
    frame = ox.features.features_from_xml(osm, tags=tags)
    return gpd.clip(frame, box(*BBOX))


def treat_kind(row: object) -> str:
    amenity = getattr(row, "amenity", None)
    shop = getattr(row, "shop", None)
    if amenity == "ice_cream":
        return "ice cream"
    if amenity == "cafe":
        return "café"
    if shop == "bakery":
        return "bakery"
    return "sweets"


def build_treats(osm: Path) -> gpd.GeoDataFrame:
    frame = fetch(osm, TREATS).reset_index()
    frame["geometry"] = frame.geometry.representative_point()
    frame["kind"] = [treat_kind(row) for row in frame.itertuples()]
    frame["name"] = frame["name"].fillna("") if "name" in frame else ""
    out = frame[["kind", "name", "geometry"]].copy()
    out = out.sort_values(["kind", "name"]).reset_index(drop=True)
    out.insert(0, "id", range(len(out)))
    return out


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    osm = local_osm()
    built = datetime.now(UTC).isoformat(timespec="seconds")

    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as tmp:
        sources = []
        for name, tags in LAYERS.items():
            frame = fetch(osm, tags)
            if name != "roads":
                frame = frame[frame.geom_type.isin(["Polygon", "MultiPolygon"])]
            else:
                frame = frame[frame.geom_type.isin(["LineString", "MultiLineString"])]
            columns = [c for c in KEEP[name] if c in frame.columns]
            frame = frame[[*columns, "geometry"]].reset_index(drop=True)
            counts[name] = len(frame)
            path = Path(tmp) / f"{name}.geojson"
            frame.to_file(path, driver="GeoJSON")
            sources.append(f"--named-layer={name}:{path}")
            print(f"{name}: {len(frame)} features", file=sys.stderr)

        pmtiles = MAP_DIR / "warsaw.pmtiles"
        subprocess.run(
            [
                "tippecanoe",
                "--force",
                f"--output={pmtiles}",
                "--minimum-zoom=12",
                "--maximum-zoom=16",
                "--drop-densest-as-needed",
                "--simplification=4",
                "--quiet",
                *sources,
            ],
            check=True,
        )

    treats = build_treats(osm)
    counts["treats"] = len(treats)
    treats_path = DATA_DIR / "treats.geojson"
    treats.to_file(treats_path, driver="GeoJSON")

    size = pmtiles.stat().st_size
    if size > MAX_PMTILES_BYTES:
        raise SystemExit(f"{pmtiles} is {size} bytes, above the 10 MB limit")
    manifest = {
        "source": f"OpenStreetMap, Geofabrik extract {EXTRACT} ({EXTRACT_URL})",
        "source_sha256": sha256(RAW / EXTRACT),
        "license": "ODbL 1.0 — © OpenStreetMap contributors",
        "bbox_wgs84": {
            "west": BBOX[0],
            "south": BBOX[1],
            "east": BBOX[2],
            "north": BBOX[3],
        },
        "built_utc": built,
        "layers": {"pmtiles": list(LAYERS), "treat_tags": TREATS},
        "counts": counts,
        "versions": {"osmnx": ox.__version__, "geopandas": gpd.__version__},
        "tools": "osmium extract --strategy=smart; tippecanoe -Z12 -z16",
        "outputs": {
            "app/public/map/warsaw.pmtiles": {"bytes": size, "sha256": sha256(pmtiles)},
            "app/public/data/treats.geojson": {
                "bytes": treats_path.stat().st_size,
                "sha256": sha256(treats_path),
            },
        },
    }
    (MAP_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(counts), f"pmtiles {size / 1e6:.2f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
