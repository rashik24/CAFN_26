# app.py
import io
import math
import os
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import pydeck as pdk
from shapely.geometry import Point
from opencage.geocoder import OpenCageGeocode

try:
    import geopandas as gpd
except ImportError:
    import geopandas_lite as gpd

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="CAFN Food Finder Builder", page_icon="📍", layout="wide")
st.title("CAFN Food Finder")

# ============================================================
# CONFIG
# ============================================================
MAPBOX_URL = "https://api.mapbox.com/search/geocode/v6/forward"
HOURS_CSV = "cafn_hourly.csv"  # fixed/static file kept with app
ROAD_FACTOR_DEFAULT = 1.25
AVERAGE_SPEED_DEFAULT = 30.0

# Prefer secrets; environment variables are fallback.
MAPBOX_TOKEN = st.secrets.get("MAPBOX_API_KEY", os.getenv("MAPBOX_API_KEY", ""))
OPENCAGE_API_KEY = st.secrets.get("OPENCAGE_API_KEY", os.getenv("OPENCAGE_API_KEY", ""))

filter1_desc = {
    "Brown Bag": "Pre-packed grocery bags distributed to individuals or families.",
    "Food Distribution": "General food distribution events providing groceries.",
    "Food Distribution (Pickup)": "Scheduled pickup-based food distribution.",
    "Mobile Food Distribution": "Food distribution at rotating or temporary locations.",
    "Other": "Additional services not categorized elsewhere.",
    "Young adult programs": "Programs specifically designed for young adults.",
    "Youth Programs": "Programs supporting children and teenagers.",
}

filter2_desc = {
    "Feeding Program": "Programs providing prepared meals.",
    "Feeding Programs": "Multiple or recurring feeding services.",
    "Food Pantries": "Locations where groceries are distributed for home use.",
    "Shelters": "Facilities providing temporary housing and meals.",
    "Soup Kitchen": "Locations serving hot, ready-to-eat meals.",
}

# ============================================================
# HELPERS
# ============================================================
def read_table(uploaded_file):
    ext = uploaded_file.name.lower().split(".")[-1]
    if ext == "csv":
        return pd.read_csv(uploaded_file)
    if ext in ["xlsx", "xls"]:
        return pd.read_excel(uploaded_file)
    raise ValueError("Upload a CSV, XLSX, or XLS file.")


def clean_text(series):
    return series.fillna("").astype(str).str.strip()


def clean_geoid(series):
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(11)


def clean_address(address):
    if pd.isna(address):
        return ""
    return " ".join(str(address).strip().split())


def geocode_address(address, token, country="US"):
    address = clean_address(address)
    if not address:
        return {"Latitude": None, "Longitude": None, "Matched Address": None,
                "Geocode Status": "MISSING_ADDRESS", "Geocode Error": None}

    params = {"q": address, "access_token": token, "limit": 1, "autocomplete": "false"}
    if country.strip():
        params["country"] = country.strip().upper()

    try:
        response = requests.get(MAPBOX_URL, params=params, timeout=30)
        data = response.json()
        if response.status_code != 200:
            return {"Latitude": None, "Longitude": None, "Matched Address": None,
                    "Geocode Status": f"HTTP_{response.status_code}",
                    "Geocode Error": data.get("message", response.text)}
        features = data.get("features", [])
        if not features:
            return {"Latitude": None, "Longitude": None, "Matched Address": None,
                    "Geocode Status": "ZERO_RESULTS", "Geocode Error": None}
        first = features[0]
        coords = first.get("geometry", {}).get("coordinates", [])
        if len(coords) < 2:
            return {"Latitude": None, "Longitude": None, "Matched Address": None,
                    "Geocode Status": "INVALID_COORDINATES", "Geocode Error": "No coordinates returned."}
        lon, lat = coords[0], coords[1]
        props = first.get("properties", {})
        matched = props.get("full_address") or props.get("name") or first.get("place_name")
        return {"Latitude": lat, "Longitude": lon, "Matched Address": matched,
                "Geocode Status": "OK", "Geocode Error": None}
    except requests.exceptions.Timeout:
        return {"Latitude": None, "Longitude": None, "Matched Address": None,
                "Geocode Status": "TIMEOUT", "Geocode Error": "Mapbox request timed out."}
    except Exception as e:
        return {"Latitude": None, "Longitude": None, "Matched Address": None,
                "Geocode Status": "ERROR", "Geocode Error": str(e)}


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.7613
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    a = min(1.0, max(0.0, a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return r * c


def build_approx_odm(tracts_df, agencies_df, road_factor, speed_mph, tract_id_col,
                     tract_lat_col, tract_lon_col, progress=None, status=None):
    tracts = tracts_df[[tract_id_col, tract_lat_col, tract_lon_col]].copy()
    tracts.columns = ["GEOID", "TRACT_LAT", "TRACT_LON"]
    tracts["GEOID"] = clean_geoid(tracts["GEOID"])
    tracts["TRACT_LAT"] = pd.to_numeric(tracts["TRACT_LAT"], errors="coerce")
    tracts["TRACT_LON"] = pd.to_numeric(tracts["TRACT_LON"], errors="coerce")
    tracts = tracts.dropna().drop_duplicates("GEOID")

    agencies = agencies_df.copy()
    agencies["agency no."] = clean_text(agencies["agency no."])
    agencies["agency name"] = clean_text(agencies["agency name"])
    agencies["latitude"] = pd.to_numeric(agencies["latitude"], errors="coerce")
    agencies["longitude"] = pd.to_numeric(agencies["longitude"], errors="coerce")
    agencies = agencies.dropna(subset=["latitude", "longitude"])
    agencies = agencies[agencies["agency no."] != ""]

    if agencies["agency no."].duplicated().any():
        dup = agencies.loc[agencies["agency no."].duplicated(False), "agency no."].unique()[:10]
        raise ValueError(f"Agency identifier is not unique. Examples: {list(dup)}")

    rows = []
    total = len(tracts)
    for pos, (_, tract) in enumerate(tracts.iterrows(), 1):
        if status is not None:
            status.write(f"Building ODM: tract {pos:,} of {total:,} ({tract['GEOID']})")
        for _, ag in agencies.iterrows():
            geo_miles = haversine_miles(float(tract["TRACT_LAT"]), float(tract["TRACT_LON"]),
                                        float(ag["latitude"]), float(ag["longitude"]))
            miles = geo_miles * road_factor
            minutes = (miles / speed_mph) * 60.0
            row = {
                "geoid": tract["GEOID"],
                "agency no.": ag["agency no."],
                "agency name": ag["agency name"],
                "total_traveltime": round(minutes, 3),
                "total_miles": round(miles, 3),
                "distance_meters": round(miles * 1609.344, 3),
                "latitude": ag["latitude"],
                "longitude": ag["longitude"],
                "geodesic_distance_miles": round(geo_miles, 3),
                "road_factor": road_factor,
                "average_speed_mph": speed_mph,
            }
            # Carry all agency attributes into ODM
            for c in agencies.columns:
                if c not in row:
                    row[c] = ag[c]
            rows.append(row)
        if progress is not None:
            progress.progress(pos / total)
    return pd.DataFrame(rows)


def load_uploaded_geometry(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".zip":
        tmpdir = tempfile.mkdtemp(prefix="tracts_")
        zpath = Path(tmpdir) / uploaded_file.name
        zpath.write_bytes(uploaded_file.getvalue())
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(tmpdir)
        shp_files = list(Path(tmpdir).rglob("*.shp"))
        if not shp_files:
            raise ValueError("ZIP does not contain a .shp file.")
        gdf = gpd.read_file(shp_files[0])
    elif suffix in [".geojson", ".json", ".gpkg"]:
        tmpdir = tempfile.mkdtemp(prefix="tracts_")
        fpath = Path(tmpdir) / uploaded_file.name
        fpath.write_bytes(uploaded_file.getvalue())
        gdf = gpd.read_file(fpath)
    else:
        raise ValueError("Upload a ZIP shapefile, GeoJSON, or GPKG.")

    if gdf.crs is None:
        raise ValueError("Tract geometry has no CRS information.")
    if str(gdf.crs).lower() != "epsg:4326":
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def normalize_hours(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    if "day" not in df.columns:
        raise ValueError("Hourly CSV must contain a 'day' column.")
    df = df[df["day"].astype(str).str.strip() != "Ist"]
    df["day"] = df["day"].astype(str).str.strip().str.title()
    if "agency no." in df.columns:
        df["agency_key"] = clean_text(df["agency no."])
    elif "name" in df.columns:
        df["agency_key"] = clean_text(df["name"])
    elif "agency" in df.columns:
        df["agency_key"] = clean_text(df["agency"])
    else:
        raise ValueError("Hourly CSV needs one of: 'agency no.', 'name', or 'agency'.")
    return df

# ============================================================
# STEP 1 — AGENCY UPLOAD + GEOCODING
# ============================================================
st.header("1. Upload and Geocode Agencies")
agency_upload = st.file_uploader("Upload agency master file", type=["csv", "xlsx", "xls"], key="agency_upload")

if agency_upload is not None:
    try:
        raw_agencies = read_table(agency_upload)
        st.dataframe(raw_agencies.head(25), use_container_width=True)

        cols = list(raw_agencies.columns)
        id_choice = st.selectbox("Unique agency identifier", cols, key="id_choice")
        name_choice = st.selectbox("Agency/site name", cols,
                                   index=cols.index("Site Name") if "Site Name" in cols else 0,
                                   key="name_choice")
        address_choice = st.selectbox("Full address column for geocoding", cols,
                                      index=cols.index("Address") if "Address" in cols else 0,
                                      key="address_choice")
        country = st.text_input("Country code", value="US")
        request_delay = st.number_input("Delay between geocoding requests", 0.0, 5.0, 0.1, 0.1)

        work = raw_agencies.copy()
        work["agency no."] = clean_text(work[id_choice])
        work["agency name"] = clean_text(work[name_choice])

        dup_count = int(work["agency no."].duplicated(keep=False).sum())
        missing_count = int((work["agency no."] == "").sum())
        if missing_count:
            st.error(f"Selected identifier has {missing_count:,} missing values.")
        if dup_count:
            st.error(f"Selected identifier has {dup_count:,} rows involved in duplicates.")
        if not missing_count and not dup_count:
            st.success("Selected agency identifier is unique.")

        run_geo = st.button("Geocode agencies", type="primary", disabled=bool(missing_count or dup_count))
        if run_geo:
            if not MAPBOX_TOKEN:
                st.error("Set MAPBOX_API_KEY in Streamlit secrets or environment variables.")
            else:
                result = work.copy()
                for c in ["Latitude", "Longitude", "Matched Address", "Geocode Status", "Geocode Error"]:
                    if c not in result.columns:
                        result[c] = None
                pb = st.progress(0)
                msg = st.empty()
                for pos, (idx, row) in enumerate(result.iterrows(), 1):
                    addr = clean_address(row[address_choice])
                    msg.write(f"Geocoding {pos:,} of {len(result):,}: {addr}")
                    lat = pd.to_numeric(pd.Series([row.get("Latitude")]), errors="coerce").iloc[0]
                    lon = pd.to_numeric(pd.Series([row.get("Longitude")]), errors="coerce").iloc[0]
                    if pd.isna(lat) or pd.isna(lon):
                        out = geocode_address(addr, MAPBOX_TOKEN, country)
                        for c, v in out.items():
                            result.at[idx, c] = v
                        if request_delay > 0 and pos < len(result):
                            time.sleep(request_delay)
                    pb.progress(pos / len(result))
                pb.empty(); msg.empty()
                result = result.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
                st.session_state["agencies_geocoded"] = result
    except Exception as e:
        st.error(f"Agency setup failed: {e}")

if "agencies_geocoded" in st.session_state:
    geocoded = st.session_state["agencies_geocoded"]
    st.subheader("Geocoded agencies")
    st.dataframe(geocoded.head(100), use_container_width=True)
    st.download_button("Download geocoded agencies", geocoded.to_csv(index=False).encode("utf-8-sig"),
                       "geocoded_agencies.csv", "text/csv")

# ============================================================
# STEP 2 — UPLOAD GEOID / TRACT COORDINATE SOURCE
# ============================================================

st.header("2. Upload GEOID Source")

st.caption(
    "Upload the Census tract coordinate file used to build the ODM. "
    "The file must contain a GEOID and tract latitude/longitude coordinates."
)

tract_centroid_upload = st.file_uploader(
    "Upload GEOID / tract-coordinate source",
    type=["csv", "xlsx", "xls"],
    key="tract_centroid",
)

if tract_centroid_upload is not None:

    try:
        tract_source = read_table(
            tract_centroid_upload
        )

        st.subheader("GEOID Source Preview")

        st.dataframe(
            tract_source.head(25),
            use_container_width=True,
        )

        tc = list(tract_source.columns)

        # ----------------------------------------------------
        # GEOID column
        # ----------------------------------------------------

        geoid_default = (
            tc.index("GEOID")
            if "GEOID" in tc
            else 0
        )

        geoid_col = st.selectbox(
            "GEOID column",
            tc,
            index=geoid_default,
            key="tract_geoid_col",
        )

        # ----------------------------------------------------
        # Detect likely latitude column
        # ----------------------------------------------------

        lat_candidates = [
            c for c in tc
            if str(c).strip().lower()
            in [
                "ycoord",
                "latitude",
                "lat",
                "tract_lat_col",
                "tract_lat",
            ]
        ]

        lat_default = (
            tc.index(lat_candidates[0])
            if lat_candidates
            else 0
        )

        tract_lat_col = st.selectbox(
            "Tract latitude column",
            tc,
            index=lat_default,
            key="tract_lat_col_select",
        )

        # ----------------------------------------------------
        # Detect likely longitude column
        # ----------------------------------------------------

        lon_candidates = [
            c for c in tc
            if str(c).strip().lower()
            in [
                "xcoord",
                "longitude",
                "lon",
                "lng",
                "tract_lng_col",
                "tract_lon",
            ]
        ]

        lon_default = (
            tc.index(lon_candidates[0])
            if lon_candidates
            else 0
        )

        tract_lon_col = st.selectbox(
            "Tract longitude column",
            tc,
            index=lon_default,
            key="tract_lon_col_select",
        )

        # ----------------------------------------------------
        # Standardize
        # ----------------------------------------------------

        tracts_df = tract_source[
            [
                geoid_col,
                tract_lat_col,
                tract_lon_col,
            ]
        ].copy()

        tracts_df.columns = [
            "GEOID",
            "TRACT_LAT_COL",
            "TRACT_LNG_COL",
        ]

        tracts_df["GEOID"] = clean_geoid(
            tracts_df["GEOID"]
        )

        tracts_df["TRACT_LAT_COL"] = pd.to_numeric(
            tracts_df["TRACT_LAT_COL"],
            errors="coerce",
        )

        tracts_df["TRACT_LNG_COL"] = pd.to_numeric(
            tracts_df["TRACT_LNG_COL"],
            errors="coerce",
        )

        tracts_df = tracts_df.dropna(
            subset=[
                "GEOID",
                "TRACT_LAT_COL",
                "TRACT_LNG_COL",
            ]
        )

        tracts_df = tracts_df.drop_duplicates(
            subset=["GEOID"],
            keep="first",
        )

        # Save for Step 3
        st.session_state["tracts_df"] = tracts_df

        st.success(
            f"GEOID source ready: "
            f"{len(tracts_df):,} valid Census tracts."
        )

        st.dataframe(
            tracts_df.head(25),
            use_container_width=True,
        )

    except Exception as e:

        st.error(
            f"Could not prepare GEOID source: {e}"
        )

# ============================================================
# STEP 3 — GENERATE ODM
# ============================================================

st.header("3. Generate ODM")

if "agencies_geocoded" not in st.session_state:

    st.info(
        "Geocode the agency file in Step 1 first."
    )

elif "tracts_df" not in st.session_state:

    st.info(
        "Upload and prepare the GEOID source in Step 2 first."
    )

else:

    agencies_df = (
        st.session_state[
            "agencies_geocoded"
        ].copy()
    )

    tracts_df = (
        st.session_state[
            "tracts_df"
        ].copy()
    )

    # --------------------------------------------------------
    # ODM settings
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        road_factor = st.number_input(
            "Road-distance factor",
            min_value=1.0,
            max_value=3.0,
            value=ROAD_FACTOR_DEFAULT,
            step=0.05,
            key="odm_road_factor",
        )

    with col2:

        average_speed = st.number_input(
            "Average speed (mph)",
            min_value=5.0,
            max_value=80.0,
            value=AVERAGE_SPEED_DEFAULT,
            step=1.0,
            key="odm_average_speed",
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    valid_agencies = agencies_df.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Geocoded agencies",
        f"{len(valid_agencies):,}",
    )

    col2.metric(
        "Census tracts",
        f"{len(tracts_df):,}",
    )

    col3.metric(
        "Expected ODM rows",
        f"{len(valid_agencies) * len(tracts_df):,}",
    )

    # --------------------------------------------------------
    # Generate ODM
    # --------------------------------------------------------

    generate_odm = st.button(
        "Generate ODM",
        type="primary",
        use_container_width=True,
        key="generate_odm_button",
    )

    if generate_odm:

        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        try:

            odm_df = build_approx_odm(
                tracts_df=tracts_df,
                agencies_df=agencies_df,
                road_factor=road_factor,
                speed_mph=average_speed,
                tract_id_col="GEOID",
                tract_lat_col="TRACT_LAT_COL",
                tract_lon_col="TRACT_LNG_COL",
                progress=progress_bar,
                status=status_placeholder,
            )

            # Save for final Food Finder
            st.session_state["odm_df"] = odm_df

            progress_bar.empty()
            status_placeholder.empty()

            st.success(
                f"ODM generated successfully: "
                f"{len(odm_df):,} tract-agency records."
            )

        except Exception as e:

            progress_bar.empty()
            status_placeholder.empty()

            st.error(
                f"ODM generation failed: {e}"
            )


# ------------------------------------------------------------
# ODM preview + download
# ------------------------------------------------------------

if "odm_df" in st.session_state:

    odm_df = st.session_state["odm_df"]

    st.subheader("Generated ODM")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "ODM rows",
        f"{len(odm_df):,}",
    )

    col2.metric(
        "Unique GEOIDs",
        f"{odm_df['geoid'].nunique():,}",
    )

    col3.metric(
        "Unique agencies",
        f"{odm_df['agency no.'].nunique():,}",
    )

    st.dataframe(
        odm_df.head(250),
        use_container_width=True,
        hide_index=True,
    )

    odm_csv = (
        odm_df
        .to_csv(index=False)
        .encode("utf-8-sig")
    )

    st.download_button(
        "Download ODM",
        data=odm_csv,
        file_name="ODM_CAFN_generated.csv",
        mime="text/csv",
        use_container_width=True,
    )
# ============================================================
# STEP 4 — LOAD STATIC FOOD FINDER DATA
# ============================================================

st.header("4. Load Static Food Finder Data")


# ------------------------------------------------------------
# 4A — Operating hours
# ------------------------------------------------------------

st.subheader("Operating Hours")

try:

    hourly_df = normalize_hours(
        pd.read_csv(
            HOURS_CSV
        )
    )

    st.session_state[
        "hourly_df"
    ] = hourly_df

    st.success(
        f"Loaded {HOURS_CSV}: "
        f"{len(hourly_df):,} operating-hour records."
    )

except Exception as e:

    hourly_df = None

    st.error(
        f"Could not load static hourly file "
        f"'{HOURS_CSV}': {e}"
    )


# ------------------------------------------------------------
# 4B — Census tract polygons
# ------------------------------------------------------------

st.subheader("Census Tract Geometry")

TRACT_GEOMETRY_FILE = (
    "cb_2023_37_tract_500k.shp"
)

try:

    tracts_gdf = gpd.read_file(
        TRACT_GEOMETRY_FILE
    )

    # Ensure geographic coordinates
    if tracts_gdf.crs is None:

        raise ValueError(
            "Census tract geometry has no CRS information."
        )

    if (
        str(tracts_gdf.crs).lower()
        != "epsg:4326"
    ):

        tracts_gdf = tracts_gdf.to_crs(
            epsg=4326
        )

    # Must have GEOID
    if "GEOID" not in tracts_gdf.columns:

        raise ValueError(
            "Census tract geometry does not contain a GEOID column."
        )

    tracts_gdf["GEOID_STD"] = clean_geoid(
        tracts_gdf["GEOID"]
    )

    # Build spatial index
    _ = tracts_gdf.sindex

    # Save for final Food Finder
    st.session_state[
        "tracts_gdf"
    ] = tracts_gdf

    st.success(
        f"Loaded Census tract geometry: "
        f"{len(tracts_gdf):,} tract polygons."
    )

except Exception as e:

    st.error(
        f"Could not load static Census tract geometry "
        f"'{TRACT_GEOMETRY_FILE}': {e}"
    )




# ============================================================
# STEP 5 — FINAL FOOD FINDER
# ============================================================
st.header("5. Food Finder")
ready = all(k in st.session_state for k in ["agencies_geocoded", "odm_df", "tracts_gdf"]) and hourly_df is not None

if not ready:
    st.info("Complete Steps 1–4 to activate the Food Finder.")
    st.stop()

agencies = st.session_state["agencies_geocoded"].copy()
odm_df = st.session_state["odm_df"].copy()
tracts_gdf = st.session_state["tracts_gdf"]

# Normalize
odm_df.columns = odm_df.columns.str.strip().str.lower()
agencies.columns = agencies.columns.str.strip().str.lower()
odm_df["geoid"] = clean_geoid(odm_df["geoid"])
odm_df["agency no."] = clean_text(odm_df["agency no."])
odm_df["agency name"] = clean_text(odm_df["agency name"])

# Merge useful master attributes into ODM when they are not already present
master_fields = [c for c in ["agency no.", "hispanic", "county", "contact", "operating hours",
                              "filter_1", "filter_2", "choice", "zip code"] if c in agencies.columns]
if len(master_fields) > 1:
    supplement = agencies[master_fields].drop_duplicates("agency no.")
    missing_fields = [c for c in master_fields if c != "agency no." and c not in odm_df.columns]
    if missing_fields:
        odm_df = odm_df.merge(supplement[["agency no."] + missing_fields], on="agency no.", how="left")

mode = st.radio("Choose input mode:", ["Address", "Zip Code"])
user_lat = user_lon = user_geoid = None

if mode == "Address":
    user_address = st.text_input("Enter your address (e.g., 123 Main St, Raleigh, NC):")
    if user_address:
        if not OPENCAGE_API_KEY:
            st.error("Set OPENCAGE_API_KEY in Streamlit secrets or environment variables.")
            st.stop()
        geocoder = OpenCageGeocode(OPENCAGE_API_KEY)
        try:
            results = geocoder.geocode(user_address)
            if not results:
                st.error("Could not geocode your address."); st.stop()
            user_lat = results[0]["geometry"]["lat"]
            user_lon = results[0]["geometry"]["lng"]
            point = Point(user_lon, user_lat)
            matched = tracts_gdf[tracts_gdf.contains(point)]
            if matched.empty:
                st.error("Could not match your location to a census tract."); st.stop()
            user_geoid = matched.iloc[0]["GEOID_STD"]
            st.write("GEOID", user_geoid)
        except Exception as e:
            st.error(f"Geocoding error: {e}"); st.stop()
else:
    zip_code = st.text_input("Enter your ZIP code:")
    if zip_code and "zip" in odm_df.columns:
        subset = odm_df[odm_df["zip"].astype(str).str.strip() == zip_code.strip()]
        if subset.empty:
            st.warning("No agencies found in that ZIP code."); st.stop()
        odm_df = subset.drop_duplicates(subset=["agency name"])

user_threshold = st.number_input("Enter travel time threshold (minutes):", min_value=5, max_value=120,
                                 value=20, step=5)

if user_geoid is not None:
    agencies_nearby = odm_df[(odm_df["geoid"] == user_geoid) &
                             (pd.to_numeric(odm_df["total_traveltime"], errors="coerce") <= user_threshold)]
    if agencies_nearby.empty:
        st.warning(f"No agencies within {user_threshold} minutes; expanding search by 40 minutes.")
        agencies_nearby = odm_df[pd.to_numeric(odm_df["total_traveltime"], errors="coerce") <= user_threshold + 40]
else:
    agencies_nearby = odm_df.copy()

df = agencies_nearby.copy()

show_choice_only = st.checkbox("Show only Choice Pantries", value=False)
if "filter_1" in df.columns:
    st.markdown("### Select Categories")
    vals = sorted(df["filter_1"].dropna().astype(str).unique())
    sel1 = st.multiselect("", vals, label_visibility="collapsed", key="filter_1_multi")
    for val in sel1:
        st.info(f"**{val}**: {filter1_desc.get(val, 'No description available.')}")
    filtered_df = df[df["filter_1"].isin(sel1)] if sel1 else df.copy()
else:
    filtered_df = df.copy()

if not filtered_df.empty and "filter_2" in filtered_df.columns:
    st.markdown("### Select Subcategories")
    vals2 = sorted(filtered_df["filter_2"].dropna().astype(str).unique())
    sel2 = st.multiselect("Filter 2", vals2, label_visibility="collapsed", key="filter_2_multi")
    if sel2:
        filtered_df = filtered_df[filtered_df["filter_2"].isin(sel2)]
        for val in sel2:
            st.info(f"**{val}**: {filter2_desc.get(val, 'No description available.')}")

if show_choice_only and "choice" in filtered_df.columns:
    filtered_df = filtered_df[pd.to_numeric(filtered_df["choice"], errors="coerce").fillna(0) == 1]

if "county" in filtered_df.columns:
    st.markdown("### Filter by County")
    county_vals = sorted(filtered_df["county"].dropna().astype(str).str.strip().unique())
    selected_counties = st.multiselect("Select county/counties", county_vals)
    if selected_counties:
        filtered_df = filtered_df[filtered_df["county"].astype(str).str.strip().isin(selected_counties)]

if "hispanic" in filtered_df.columns:
    st.markdown("### Language Support")
    if st.checkbox("Show only pantries that speak Spanish/Hispanic", value=False):
        filtered_df = filtered_df[pd.to_numeric(filtered_df["hispanic"], errors="coerce").fillna(0).astype(int) == 1]

st.markdown("### Filter by Operating Day")
unique_days = sorted(hourly_df["day"].dropna().unique())
selected_day = st.selectbox("Select Day", ["Any"] + unique_days)
if selected_day != "Any":
    open_keys = set(hourly_df.loc[hourly_df["day"] == selected_day, "agency_key"].dropna().astype(str).str.strip())
    # Prefer ID match if hours has agency no.; otherwise name match.
    if "agency no." in hourly_df.columns:
        filtered_df = filtered_df[filtered_df["agency no."].astype(str).str.strip().isin(open_keys)]
    else:
        filtered_df = filtered_df[filtered_df["agency name"].astype(str).str.strip().isin(open_keys)]

if filtered_df.empty:
    st.warning("No agencies found matching your filters.")
    st.stop()

if "total_traveltime" in filtered_df.columns:
    filtered_df["total_traveltime"] = pd.to_numeric(filtered_df["total_traveltime"], errors="coerce").round(2)
if "total_miles" in filtered_df.columns:
    filtered_df["total_miles"] = pd.to_numeric(filtered_df["total_miles"], errors="coerce").round(2)

cols = [c for c in ["agency name", "address", "operating hours", "contact", "total_traveltime", "total_miles"]
        if c in filtered_df.columns]
st.dataframe(filtered_df[cols].drop_duplicates().sort_values("total_traveltime" if "total_traveltime" in cols else "agency name"),
             use_container_width=True)

# MAP
user_df = pd.DataFrame(columns=["name", "latitude", "longitude", "color_r", "color_g", "color_b", "tooltip"])
if user_lat is not None and user_lon is not None:
    user_df = pd.DataFrame({"name": ["Your Location"], "latitude": [user_lat], "longitude": [user_lon],
                            "color_r": [0], "color_g": [0], "color_b": [255], "tooltip": ["Your Location"]})

map_df = filtered_df.copy()
if "latitude" in map_df.columns and "longitude" in map_df.columns:
    map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
    map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
    map_df["color_r"] = 255; map_df["color_g"] = 0; map_df["color_b"] = 0
    tt = map_df["total_traveltime"].astype(str) if "total_traveltime" in map_df.columns else ""
    tm = map_df["total_miles"].astype(str) if "total_miles" in map_df.columns else ""
    nm = map_df["agency name"].astype(str)
    map_df["tooltip"] = "Agency: " + nm
    if "total_traveltime" in map_df.columns:
        map_df["tooltip"] += "<br>Travel Time (min): " + tt
    if "total_miles" in map_df.columns:
        map_df["tooltip"] += "<br>Distance (miles): " + tm

    combined = pd.concat([user_df, map_df], ignore_index=True, sort=False)
    layer = pdk.Layer("ScatterplotLayer", combined.dropna(subset=["longitude", "latitude"]),
                      get_position='[longitude, latitude]', get_color='[color_r, color_g, color_b]',
                      get_radius=250, pickable=True)
    view_state = pdk.ViewState(longitude=user_lon if user_lon is not None else float(map_df["longitude"].mean()),
                               latitude=user_lat if user_lat is not None else float(map_df["latitude"].mean()),
                               zoom=10, pitch=0)
    deck = pdk.Deck(map_style='mapbox://styles/mapbox/light-v9', initial_view_state=view_state,
                    layers=[layer], tooltip={"html": "{tooltip}", "style": {"color": "white"}})
    st.pydeck_chart(deck)
