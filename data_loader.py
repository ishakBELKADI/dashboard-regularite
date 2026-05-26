import pandas as pd
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from google.transit import gtfs_realtime_pb2
from pathlib import Path

import pandas as pd
import time
from datetime import datetime
import re
import html

URL_ALERTS = "https://proxy.transport.data.gouv.fr/resource/sncf-gtfs-rt-service-alerts"

def debug_log(message):
    log_path = Path(__file__).resolve().parent / "debug_log.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n")


print("DATA_LOADER chargé", flush=True)
def unix_to_gtfs_time(ts: int, date_trip: str) -> str | pd.NA:
    if pd.isna(ts) or ts is None:
        return pd.NA
    try:
        dt = datetime.fromtimestamp(int(ts))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return pd.NA



DF_STATE = None
STATE_FILE = Path(__file__).resolve().parent / "df_state.xlsx"
HISTORY_FILE = Path(__file__).resolve().parent / "regularite_history.xlsx"
TRAIN_STATUS_UPCOMING = "upcoming"
TRAIN_STATUS_ACTIVE = "active"
TRAIN_STATUS_FINISHED = "finished"
TRAIN_STATUS_UNKNOWN = "unknown"


LISTE_GARES = [
    "Nancy", "Reims", "Charleville-Mézières", "Sedan",
    "Champagne-Ardenne TGV", "Meuse TGV", "Lorraine TGV",
    "Remiremont", "Sarrebourg", "Strasbourg", "Metz",
    "Thionville", "Epinal", "Lunéville", "Saverne",
    "Sélestat", "Colmar", "Mulhouse", "Paris Est", "Luxembourg"
]

LISTE_GARES_ALLEMAGNE = [
    "Sarrebruck",
    "Kaiserslautern Hbf",
    "Mannheim Hbf",
    "Francfort sur le Main",
    "Karlsruhe Hbf",
    "Stuttgart Hbf",
    "Ulm Hbf",
    "Augsburg Hbf",
    "Munich",
    "Offenburg",
    "Lahr (Schwarzw)",
    "Emmendingen",
    "Fribourg-en-Brisgau",
    "Baden-Baden",
]


def load_theoretical_data():
    stops = pd.read_csv("../Recuperation PTR/theorique/stops.txt")
    trips = pd.read_csv("../Recuperation PTR/theorique/trips.txt")
    stop_times = pd.read_csv("../Recuperation PTR/theorique/stop_times.txt")
    calendar_dates = pd.read_csv("../Recuperation PTR/theorique/calendar_dates.txt")

    date_today = int(datetime.today().strftime("%Y%m%d"))

    services_today = calendar_dates.loc[
        (calendar_dates["date"] == date_today) &
        (calendar_dates["exception_type"] == 1),
        "service_id"
    ].unique()

    trips_today = trips[trips["service_id"].isin(services_today)].copy()
    trips_today["tag"] = trips_today["trip_id"].str.extract(r":([A-Z0-9]+):FR:", expand=False)

    tgv_tags = ["OUI", "OGO", "ICE"]
    trips_tgv_today = trips_today[trips_today["tag"].isin(tgv_tags)].copy()

    stop_times_today = stop_times[stop_times["trip_id"].isin(trips_tgv_today["trip_id"])].copy()

    df_today = stop_times_today.merge(
        stops[["stop_id", "stop_name"]],
        on="stop_id",
        how="left"
    )

    df_today = df_today.merge(
        trips_tgv_today[["trip_id", "trip_headsign"]],
        on="trip_id",
        how="left"
    )

    df_today = df_today.sort_values(["trip_id", "stop_sequence"]).copy()

    df_export = df_today.copy()
    df_export["date"] = datetime.today().strftime("%Y%m%d")

    df_export = df_export[[
        "trip_id",
        "trip_headsign",
        "date",
        "stop_sequence",
        "stop_name",
        "arrival_time",
        "departure_time",
        "stop_id"
    ]].rename(columns={
        "trip_headsign": "train_number"
    })

    return df_export.sort_values(["trip_id", "stop_sequence"]).copy()

def clean_html_text(x):
    if pd.isna(x) or x is None:
        return ""

    txt = str(x)
    txt = re.sub(r"<[^>]+>", " ", txt)
    txt = html.unescape(txt)
    txt = re.sub(r"\s+", " ", txt).strip()

    return txt


def get_alert_text(translated_string):
    if not translated_string.translation:
        return ""

    for t in translated_string.translation:
        if t.language and t.language.lower().startswith("fr"):
            return t.text

    return translated_string.translation[0].text


def extract_train_number_from_trip_id(trip_id):
    if pd.isna(trip_id) or trip_id is None:
        return None

    s = str(trip_id)

    # Cas OCESN9551F, OCESN9551, etc.
    match = re.search(r"OCESN(\d+)", s)
    if match:
        return match.group(1)

    # Cas fallback
    match = re.search(r"(\d{4,6})", s)
    return match.group(1) if match else None

def get_service_alerts_by_train():
    r = requests.get(URL_ALERTS, timeout=20)
    r.raise_for_status()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)

    rows = []

    for entity in feed.entity:
        if not entity.HasField("alert"):
            continue

        alert = entity.alert

        titre = clean_html_text(get_alert_text(alert.header_text))
        description = clean_html_text(get_alert_text(alert.description_text))

        cause = ""
        if "Cause :" in description:
            cause = description.split("Cause :", 1)[1].strip()
        else:
            cause = description.strip()

        for informed in alert.informed_entity:
            trip_id = informed.trip.trip_id if informed.HasField("trip") else None
            train_number = extract_train_number_from_trip_id(trip_id)

            if train_number:
                rows.append({
                    "train_number": train_number,
                    "cause": cause,
                    "titre_alerte": titre,
                    "description_alerte": description,
                })

    df_alerts = pd.DataFrame(rows)

    if df_alerts.empty:
        return pd.DataFrame(columns=[
            "train_number",
            "cause",
            "titre_alerte",
            "description_alerte"
        ])

    df_alerts = (
        df_alerts
        .drop_duplicates(subset=["train_number", "cause", "titre_alerte"])
        .groupby("train_number", as_index=False)
        .agg(
            cause=("cause", lambda x: " | ".join([v for v in x if v])),
            titre_alerte=("titre_alerte", lambda x: " | ".join([v for v in x if v])),
            description_alerte=("description_alerte", lambda x: " | ".join([v for v in x if v])),
        )
    )

    return df_alerts


def get_rt_tripupdates(
    url="https://proxy.transport.data.gouv.fr/resource/sncf-gtfs-rt-trip-updates",
    timeout=20
):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)

    rt_by_trip = {}
    feed_date = None
    feed_ts = getattr(feed.header, "timestamp", None)

    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue

        tu = entity.trip_update
        trip_id = tu.trip.trip_id
        start_date = tu.trip.start_date
        tu_ts = tu.timestamp

        if feed_date is None and start_date:
            feed_date = start_date

        if trip_id not in rt_by_trip:
            rt_by_trip[trip_id] = {}

        for stu in tu.stop_time_update:
            schedule_relationship = stu.schedule_relationship
            stop_id = stu.stop_id

            arr = None
            dep = None

            if stu.HasField("arrival") and stu.arrival.HasField("delay"):
                arr = stu.arrival.delay

            if stu.HasField("departure") and stu.departure.HasField("delay"):
                dep = stu.departure.delay

            rt_by_trip[trip_id][stop_id] = {
                "arrival_delay": arr,
                "departure_delay": dep,
                "tu_timestamp": tu_ts,
                "start_date": start_date,
                "schedule_relationship": schedule_relationship,
            }

    return rt_by_trip, feed_ts


def gtfs_time_to_seconds(time_str):
    if not time_str or pd.isna(time_str):
        return None
    try:
        h, m, s = map(int, str(time_str).split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        return None


def apply_delay(time_str, delay_s):
    if pd.isna(time_str) or time_str is None:
        return pd.NA

    try:
        base_s = gtfs_time_to_seconds(time_str)
        if base_s is None:
            return pd.NA

        total = base_s + int(delay_s or 0)
        if total < 0:
            total = 0

        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    except Exception:
        return pd.NA


def normalize_trip_id(trip_id):
    s = str(trip_id)
    parts = s.rsplit(":", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        return parts[0]
    return s



def save_state_to_excel(df, filename="df_state.xlsx"):
    output_path = Path(__file__).resolve().parent / filename
    df.to_excel(output_path, index=False)

def save_regularite_history(metrics):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame([{
        # Timestamp
        "datetime": now_str,

        # GLOBAL
        "taux_regularite_desserte": metrics["taux_regularite_desserte"],
        "taux_regularite_terminus": metrics["taux_regularite_terminus"],
        "nb_dessertes": metrics["nb_dessertes"],
        "nb_dessertes_regulieres": metrics["nb_dessertes_regulieres"],
        "nb_trains_terminus": metrics["nb_trains_terminus"],
        "nb_trains_terminus_reguliers": metrics["nb_trains_terminus_reguliers"],

        # RR
        "taux_regularite_rr": metrics["taux_regularite_rr"],
        "nb_dessertes_rr": metrics["nb_dessertes_rr"],
        "nb_dessertes_rr_regulieres": metrics["nb_dessertes_rr_regulieres"],

        # ALEO
        "taux_regularite_aleo": metrics["taux_regularite_aleo"],
        "nb_dessertes_aleo": metrics["nb_dessertes_aleo"],
        "nb_dessertes_aleo_regulieres": metrics["nb_dessertes_aleo_regulieres"],

        # INTERSECTEURS
        "taux_regularite_intersecteurs": metrics["taux_regularite_intersecteurs"],
        "nb_dessertes_intersecteurs": metrics["nb_dessertes_intersecteurs"],
        "nb_dessertes_intersecteurs_regulieres": metrics["nb_dessertes_intersecteurs_regulieres"],

        # Terminus
        "taux_regularite_terminus_rr": metrics["taux_regularite_terminus_rr"],
        "nb_trains_terminus_rr": metrics["nb_trains_terminus_rr"],
        "nb_trains_terminus_rr_reguliers": metrics["nb_trains_terminus_rr_reguliers"],

        "taux_regularite_terminus_aleo": metrics["taux_regularite_terminus_aleo"],
        "nb_trains_terminus_aleo": metrics["nb_trains_terminus_aleo"],
        "nb_trains_terminus_aleo_reguliers": metrics["nb_trains_terminus_aleo_reguliers"],

        "taux_regularite_terminus_intersecteurs": metrics["taux_regularite_terminus_intersecteurs"],
        "nb_trains_terminus_intersecteurs": metrics["nb_trains_terminus_intersecteurs"],
        "nb_trains_terminus_intersecteurs_reguliers": metrics["nb_trains_terminus_intersecteurs_reguliers"],
    }])

    if HISTORY_FILE.exists():
        df_hist = pd.read_excel(HISTORY_FILE)

        # Sécurise les anciennes versions du fichier
        for col in new_row.columns:
            if col not in df_hist.columns:
                df_hist[col] = pd.NA

        df_hist = pd.concat([df_hist, new_row], ignore_index=True)

    else:
        df_hist = new_row

    df_hist.to_excel(HISTORY_FILE, index=False)

VOYAGEURS_FILE = Path(__file__).resolve().parent / "voyageurs.xlsx"

# =========================
# NORMALISATION ROBUSTE DES GARES
# A METTRE DANS data_loader.py
# (remplace complètement ton ancienne fonction normalize_gare_name)
# =========================

import re
import unicodedata


def normalize_gare_name(gare):
    if pd.isna(gare):
        return ""

    s = str(gare).strip().lower()

    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))

    patterns_to_remove = [
        r"batiment voyageurs",
        r"batim\. voyageurs",
        r"bâtiment voyageurs",
        r"uebri?ge empfaenger",
        r"\*00\*",
        r"\bbv\b",
        r"\bgare\b",
        r"\bstation\b",
    ]

    for pattern in patterns_to_remove:
        s = re.sub(pattern, "", s)

    replacements = {
        "-": " ",
        "/": " ",
        "'": " ",
        "’": " ",
        ".": " ",
        ",": " ",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    s = re.sub(r"\s+", " ", s).strip()

    # Maintenant seulement, on remplace St / Ste
    s = re.sub(r"\bst\b", "saint", s)
    s = re.sub(r"\bste\b", "sainte", s)

    s = re.sub(r"\s+", " ", s).strip()

    special_cases = {
        "paris est": "paris est",
        "strasbourg ville": "strasbourg",
        "nancy ville": "nancy",
        "metz ville": "metz",
        "dijon ville": "dijon",
        "mulhouse ville": "mulhouse",
        "macon ville": "macon",
        "macon loche tgv": "macon",
        "nimes": "nimes centre",
        "nimes centre": "nimes centre",
        "nimes pont du gard": "nimes centre",

        "champagne ardenne tgv": "champagne ardenne tgv",
        "lorraine tgv": "lorraine tgv",
        "meuse tgv": "meuse tgv",

        "frankfurt main hbf": "francfort sur le main",
        "mannheim": "mannheim hbf",
        "saarbruecken hbf": "sarrebruck",
        "munchen hbf munich hbf": "munich",

        "ringsheim europa park": "ringsheim",
        "valence tgv": "valence tgv rhone alpes sud",
        "roissy aeroport cdg 2 tgv": "aeroport charles de gaulle 2 tgv",
    }

    if s in special_cases:
        return special_cases[s]

    return s

def enrich_with_voyageurs_descendants(df_state):
    df_state = df_state.copy()

    if "voyageurs_descendants" not in df_state.columns:
        df_state["voyageurs_descendants"] = 0

    if not VOYAGEURS_FILE.exists():
        debug_log("Fichier voyageurs.xlsx introuvable")
        return df_state

    df_voy = pd.read_excel(VOYAGEURS_FILE)

    colonnes_obligatoires = ["numero_train", "des_lib_ext", "nb_voy_desc"]
    for col in colonnes_obligatoires:
        if col not in df_voy.columns:
            debug_log(f"Colonne manquante dans voyageurs.xlsx : {col}")
            return df_state

    df_state["train_key"] = df_state["train_number"].astype(str).str.strip()
    df_state["gare_key"] = df_state["stop_name"].apply(normalize_gare_name)

    df_voy["train_key"] = df_voy["numero_train"].astype(str).str.strip()
    df_voy["gare_key"] = df_voy["des_lib_ext"].apply(normalize_gare_name)
    debug_log(
    df_voy[
        df_voy["numero_train"].astype(str).str.strip().eq("12142")
    ][["numero_train", "des_lib_ext", "nb_voy_desc", "gare_key"]].to_string()
    )

    df_voy["voyageurs_descendants"] = pd.to_numeric(
        df_voy["nb_voy_desc"],
        errors="coerce"
    ).fillna(0).astype(int)

    df_voy_clean = (
        df_voy[["train_key", "gare_key", "voyageurs_descendants"]]
        .groupby(["train_key", "gare_key"], as_index=False)
        .agg(voyageurs_descendants=("voyageurs_descendants", "sum"))
    )

    df_state = df_state.merge(
        df_voy_clean,
        on=["train_key", "gare_key"],
        how="left",
        suffixes=("", "_voy")
    )

    df_state["voyageurs_descendants"] = (
        df_state["voyageurs_descendants_voy"]
        .fillna(df_state["voyageurs_descendants"])
        .fillna(0)
        .astype(int)
    )

    df_state = df_state.drop(
        columns=[
            c for c in [
                "train_key",
                "gare_key",
                "voyageurs_descendants_voy"
            ]
            if c in df_state.columns
        ]
    )

    return df_state


def migrate_existing_state_with_voyageurs(force=True):
    if not STATE_FILE.exists():
        return

    df_state = pd.read_excel(STATE_FILE)

    if force:
        df_state["voyageurs_descendants"] = 0
    elif "voyageurs_descendants" in df_state.columns:
        return

    df_state = enrich_with_voyageurs_descendants(df_state)

    df_state.to_excel(STATE_FILE, index=False)
    debug_log("Migration df_state.xlsx : voyageurs_descendants recalculé.")

DF_THEORIQUE = load_theoretical_data()
migrate_existing_state_with_voyageurs(force=True)

def compute_regularite_metrics(df):
    df = df.copy()

    SEUIL = 359
    now_dt = datetime.now()

    def gtfs_to_datetime(date_str, time_str):
        if pd.isna(date_str) or pd.isna(time_str) or not date_str or not time_str:
            return pd.NaT
        try:
            date_str = str(date_str)
            h, m, s = map(int, str(time_str).split(":"))
            base = datetime.strptime(date_str, "%Y%m%d")
            return base + pd.Timedelta(hours=h, minutes=m, seconds=s)
        except Exception:
            return pd.NaT

    df["arrival_dt_real"] = df.apply(
        lambda r: gtfs_to_datetime(
            r.get("date_trip", r.get("date")),
            r.get("arrival_real", r.get("arrival_time"))
        ),
        axis=1
    )

    df["desserte_traversee"] = (
        df["arrival_dt_real"].notna()
        & (df["arrival_dt_real"] <= now_dt)
    )
    def normalize_name(x):
        if pd.isna(x):
            return ""
        return str(x).lower().strip()

    gares_allemagne_norm = [normalize_name(g) for g in LISTE_GARES_ALLEMAGNE]

    def is_gare_allemande(stop_name):
        stop = normalize_name(stop_name)
        return any(gare in stop or stop in gare for gare in gares_allemagne_norm)


    def prepare_desserte_metier(df_source):
        rows = []

        for trip_id, g in df_source.groupby("trip_id"):
            g = g.sort_values("stop_sequence").copy()
            g["is_allemagne"] = g["stop_name"].apply(is_gare_allemande)

            # Train sans Allemagne
            if not g["is_allemagne"].any():
                g["retard_metier_desserte_s"] = g["retard_s"]
                rows.append(g)
                continue

            # On garde uniquement les gares FR
            g_fr = g[~g["is_allemagne"]].copy()

            if g_fr.empty:
                continue

            first_is_de = bool(g.iloc[0]["is_allemagne"])
            last_is_de = bool(g.iloc[-1]["is_allemagne"])

            # Allemagne -> France
            # On neutralise le retard d'entrée en France
            if first_is_de and not last_is_de:

                retard_entree_france = g_fr.iloc[0]["retard_s"]

                if pd.isna(retard_entree_france):
                    retard_entree_france = 0

                g_fr["retard_metier_desserte_s"] = (
                    g_fr["retard_s"].fillna(0)
                    - retard_entree_france
                ).clip(lower=0)

                rows.append(g_fr)
                continue

            # France -> Allemagne
            # Les retards en Allemagne ne comptent pas
            if not first_is_de and last_is_de:

                g_fr["retard_metier_desserte_s"] = g_fr["retard_s"]

                rows.append(g_fr)
                continue

            # Cas atypique
            g_fr["retard_metier_desserte_s"] = g_fr["retard_s"]
            rows.append(g_fr)

        if not rows:
            return pd.DataFrame()

        return pd.concat(rows, ignore_index=True)


    df_metier = prepare_desserte_metier(df)

    df_metier["desserte_reguliere"] = (
        df_metier["retard_metier_desserte_s"] <= SEUIL
    )

    df_metier["voyageurs_descendants"] = pd.to_numeric(
        df_metier.get("voyageurs_descendants", 0),
        errors="coerce"
    ).fillna(0)

    df_metier["voyageurs_reguliers"] = df_metier["voyageurs_descendants"].where(
        df_metier["desserte_reguliere"],
        0
    )

    trains_rr = (
        df_metier.groupby("trip_id")["stop_name"]
        .apply(lambda gares: gares.isin(LISTE_GARES).all())
    )

    trip_ids_rr = trains_rr[trains_rr].index

    trains_aleo = (
        df.groupby("trip_id")["stop_name"]
        .apply(lambda gares: gares.apply(is_gare_allemande).any())
    )

    trip_ids_aleo = trains_aleo[trains_aleo].index

    trip_ids_rr = set(trip_ids_rr)
    trip_ids_aleo = set(trip_ids_aleo)

    trip_ids_intersecteurs = (
        set(df_metier["trip_id"].unique())
        - trip_ids_rr
        - trip_ids_aleo
    )

    df_rr = df_metier[df_metier["trip_id"].isin(trip_ids_rr)].copy()

    df_aleo = df_metier[df_metier["trip_id"].isin(trip_ids_aleo)].copy()

    df_intersecteurs = df_metier[
        df_metier["trip_id"].isin(trip_ids_intersecteurs)
    ].copy()


    def compute_desserte_categorie(df_cat):

        df_desserte_cat = df_cat[
            (df_cat["train_status"] == TRAIN_STATUS_FINISHED)
            |
            (
                (df_cat["train_status"] == TRAIN_STATUS_ACTIVE)
                & (df_cat["desserte_traversee"])
            )
        ].copy()

        nb_dessertes = len(df_desserte_cat)

        nb_dessertes_regulieres = int(
            df_desserte_cat["desserte_reguliere"].sum()
        )

        voyageurs_total = df_desserte_cat["voyageurs_descendants"].sum()

        voyageurs_reguliers = df_desserte_cat[
            "voyageurs_reguliers"
        ].sum()

        taux = (
            voyageurs_reguliers / voyageurs_total * 100
            if voyageurs_total > 0
            else 0
        )

        return (
            nb_dessertes,
            nb_dessertes_regulieres,
            round(taux, 2)
        )


    nb_dessertes, nb_dessertes_regulieres, taux_regularite_desserte = (
        compute_desserte_categorie(df_metier)
    )

    nb_dessertes_rr, nb_dessertes_rr_regulieres, taux_regularite_rr = (
        compute_desserte_categorie(df_rr)
    )

    nb_dessertes_aleo, nb_dessertes_aleo_regulieres, taux_regularite_aleo = (
        compute_desserte_categorie(df_aleo)
    )

    nb_dessertes_intersecteurs, nb_dessertes_intersecteurs_regulieres, taux_regularite_intersecteurs = (
        compute_desserte_categorie(df_intersecteurs)
    )

    def is_gare_allemande_nom(stop_name):
        stop = normalize_name(stop_name)
        return any(gare in stop or stop in gare for gare in gares_allemagne_norm)


    def compute_terminus_metier(df_source):
        df_finished = df_source[df_source["train_status"] == TRAIN_STATUS_FINISHED].copy()

        if df_finished.empty:
            return pd.DataFrame()

        rows = []

        for trip_id, g in df_finished.groupby("trip_id"):
            g = g.sort_values("stop_sequence").copy()
            g["is_allemagne"] = g["stop_name"].apply(is_gare_allemande_nom)

            if not g["is_allemagne"].any():
                terminus = g.iloc[-1].copy()
                terminus["retard_terminus_metier_s"] = terminus["retard_s"]
                rows.append(terminus)
                continue

            g_fr = g[~g["is_allemagne"]].copy()

            if g_fr.empty:
                continue

            first_is_de = bool(g.iloc[0]["is_allemagne"])
            last_is_de = bool(g.iloc[-1]["is_allemagne"])

            # France -> Allemagne
            if not first_is_de and last_is_de:
                terminus = g_fr.iloc[-1].copy()
                terminus["retard_terminus_metier_s"] = terminus["retard_s"]
                rows.append(terminus)
                continue

            # Allemagne -> France
            if first_is_de and not last_is_de:
                entree_france = g_fr.iloc[0]
                terminus = g_fr.iloc[-1].copy()

                retard_entree = entree_france["retard_s"] if pd.notna(entree_france["retard_s"]) else 0
                retard_fin = terminus["retard_s"] if pd.notna(terminus["retard_s"]) else 0

                terminus["retard_terminus_metier_s"] = max(retard_fin - retard_entree, 0)
                rows.append(terminus)
                continue

            # Cas atypique : dernière gare française
            terminus = g_fr.iloc[-1].copy()
            terminus["retard_terminus_metier_s"] = terminus["retard_s"]
            rows.append(terminus)

        terminus_df = pd.DataFrame(rows)

        if terminus_df.empty:
            return terminus_df

        terminus_df["voyageurs_descendants"] = pd.to_numeric(
            terminus_df.get("voyageurs_descendants", 0),
            errors="coerce"
        ).fillna(0)

        terminus_df["terminus_regulier"] = (
            terminus_df["retard_terminus_metier_s"] <= SEUIL
        )

        return terminus_df


    def compute_terminus_categorie(df_cat):
        terminus_cat = compute_terminus_metier(df_cat)

        if terminus_cat.empty:
            return 0, 0, 100

        nb_trains = len(terminus_cat)
        nb_reguliers = int(terminus_cat["terminus_regulier"].sum())

        voyageurs_total = terminus_cat["voyageurs_descendants"].sum()
        voyageurs_reguliers = terminus_cat.loc[
            terminus_cat["terminus_regulier"],
            "voyageurs_descendants"
        ].sum()

        taux = (
            voyageurs_reguliers / voyageurs_total * 100
            if voyageurs_total > 0
            else 0
        )

        return nb_trains, nb_reguliers, round(taux, 2)


    nb_trains_terminus, nb_trains_terminus_reguliers, taux_regularite_terminus = compute_terminus_categorie(df)

    nb_trains_terminus_rr, nb_trains_terminus_rr_reguliers, taux_regularite_terminus_rr = compute_terminus_categorie(df_rr)

    nb_trains_terminus_aleo, nb_trains_terminus_aleo_reguliers, taux_regularite_terminus_aleo = compute_terminus_categorie(df_aleo)

    nb_trains_terminus_intersecteurs, nb_trains_terminus_intersecteurs_reguliers, taux_regularite_terminus_intersecteurs = compute_terminus_categorie(df_intersecteurs)


    return {
        "nb_dessertes": nb_dessertes,
        "nb_dessertes_regulieres": nb_dessertes_regulieres,
        "taux_regularite_desserte": round(taux_regularite_desserte, 2),
        "nb_trains_terminus": nb_trains_terminus,
        "nb_trains_terminus_reguliers": nb_trains_terminus_reguliers,
        "taux_regularite_terminus": round(taux_regularite_terminus, 2),
        "nb_dessertes_aleo": nb_dessertes_aleo,
        "nb_dessertes_aleo_regulieres": nb_dessertes_aleo_regulieres,
        "taux_regularite_aleo": round(taux_regularite_aleo, 2),
        "nb_dessertes_rr": nb_dessertes_rr,
        "nb_dessertes_rr_regulieres": nb_dessertes_rr_regulieres,
        "taux_regularite_rr": round(taux_regularite_rr, 2),
        "taux_regularite_intersecteurs": round(taux_regularite_intersecteurs, 2),
        "nb_dessertes_intersecteurs": nb_dessertes_intersecteurs,
        "nb_dessertes_intersecteurs_regulieres": nb_dessertes_intersecteurs_regulieres,
        "nb_trains_terminus_rr": nb_trains_terminus_rr,
        "nb_trains_terminus_rr_reguliers": nb_trains_terminus_rr_reguliers,
        "taux_regularite_terminus_rr": taux_regularite_terminus_rr,
        "nb_trains_terminus_aleo": nb_trains_terminus_aleo,
        "nb_trains_terminus_aleo_reguliers": nb_trains_terminus_aleo_reguliers,
        "taux_regularite_terminus_aleo": taux_regularite_terminus_aleo,
        "nb_trains_terminus_intersecteurs": nb_trains_terminus_intersecteurs,
        "nb_trains_terminus_intersecteurs_reguliers": nb_trains_terminus_intersecteurs_reguliers,
        "taux_regularite_terminus_intersecteurs": taux_regularite_terminus_intersecteurs,
    }


def update_df_with_feed(df_state: pd.DataFrame, feed) -> pd.DataFrame:
    df_state = df_state.copy()

    df_state["trip_id"] = df_state["trip_id"].astype(str)
    df_state["trip_id_norm"] = df_state["trip_id"].apply(normalize_trip_id)
    df_state["stop_id"] = df_state["stop_id"].astype(str)
    df_state["stop_sequence"] = pd.to_numeric(df_state["stop_sequence"], errors="coerce")

    for u in feed.entity:
        if not u.HasField("trip_update"):
            continue

        tu = u.trip_update

        trip_id_feed = str(tu.trip.trip_id)
        trip_id_feed_norm = normalize_trip_id(trip_id_feed)

        # Matching sur trip_id normalisé, pas sur trip_id complet
        if trip_id_feed_norm not in df_state["trip_id_norm"].values:
            continue

        train_mask = df_state["trip_id_norm"] == trip_id_feed_norm
        train_rows = df_state.loc[train_mask].copy()
        if train_rows["train_status"].eq(TRAIN_STATUS_FINISHED).any():
            continue

        if train_rows.empty:
            continue

        stop_updates = list(tu.stop_time_update)
        if len(stop_updates) == 0:
            continue

        train_rows = train_rows.sort_values("stop_sequence").copy()

        current_arr_delay = None
        current_dep_delay = None

        rt_by_stop = {}

        for stop in stop_updates:
            stop_id = str(stop.stop_id)

            arr_delay = None
            dep_delay = None
            arr_time_unix = None
            dep_time_unix = None
            schedule_relationship = stop.schedule_relationship

            if stop.HasField("arrival"):
                if stop.arrival.HasField("delay"):
                    arr_delay = int(stop.arrival.delay)
                if stop.arrival.HasField("time"):
                    arr_time_unix = int(stop.arrival.time)

            if stop.HasField("departure"):
                if stop.departure.HasField("delay"):
                    dep_delay = int(stop.departure.delay)
                if stop.departure.HasField("time"):
                    dep_time_unix = int(stop.departure.time)

            rt_by_stop[stop_id] = {
                "arrival_delay": arr_delay,
                "departure_delay": dep_delay,
                "arrival_time_unix": arr_time_unix,
                "departure_time_unix": dep_time_unix,
                "schedule_relationship": schedule_relationship,
            }

        for idx, row in train_rows.iterrows():
            stop_id = str(row["stop_id"])
            rt = rt_by_stop.get(stop_id)
            df_state.loc[idx, "is_stop_deleted"] = False
            df_state.loc[idx, "stop_status"] = "scheduled"
            if rt is not None:
                if rt.get("schedule_relationship") == 1:
                    df_state.loc[idx, "is_stop_deleted"] = True
                    df_state.loc[idx, "stop_status"] = "skipped"
                if rt["arrival_delay"] is not None:
                    current_arr_delay = rt["arrival_delay"]

                if rt["departure_delay"] is not None:
                    current_dep_delay = rt["departure_delay"]

                if rt["arrival_time_unix"] is not None:
                    df_state.at[idx, "arrival_real"] = unix_to_gtfs_time(
                        rt["arrival_time_unix"], row["date_trip"]
                    )

                if rt["departure_time_unix"] is not None:
                    df_state.at[idx, "departure_real"] = unix_to_gtfs_time(
                        rt["departure_time_unix"], row["date_trip"]
                    )

            arr_delay_s = current_arr_delay if current_arr_delay is not None else 0
            dep_delay_s = current_dep_delay if current_dep_delay is not None else 0

            df_state.at[idx, "arrival_delay_s"] = arr_delay_s
            df_state.at[idx, "departure_delay_s"] = dep_delay_s

            # Régularité calculée uniquement sur le retard à l'arrivée
            df_state.at[idx, "retard_s"] = arr_delay_s
            df_state.at[idx, "retard_m"] = arr_delay_s / 60
            df_state.at[idx, "arrival_real"] = apply_delay(
                row["arrival_time"], arr_delay_s
            )
            df_state.at[idx, "departure_real"] = apply_delay(
                row["departure_time"], dep_delay_s
            )

    return df_state



def recompute_train_status(df_state: pd.DataFrame) -> pd.DataFrame:
    df_state = df_state.copy()
    now_dt = datetime.now()

    df_state["trip_id"] = df_state["trip_id"].astype(str)
    df_state["stop_sequence"] = pd.to_numeric(df_state["stop_sequence"], errors="coerce")

    for trip_id in df_state["trip_id"].unique():
        train_rows = df_state[df_state["trip_id"] == trip_id].copy()
        if train_rows["train_status"].eq(TRAIN_STATUS_FINISHED).any():
            continue

        if train_rows.empty:
            continue

        train_rows = train_rows.sort_values("stop_sequence")

        first_row = train_rows.iloc[0]
        last_row = train_rows.iloc[-1]

        start_time = first_row["departure_real"] if pd.notna(first_row["departure_real"]) else first_row["departure_time"]
        end_time = last_row["arrival_real"] if pd.notna(last_row["arrival_real"]) else last_row["arrival_time"]

        try:
            h1, m1, s1 = map(int, str(start_time).split(":"))
            h2, m2, s2 = map(int, str(end_time).split(":"))
        except Exception:
            df_state.loc[df_state["trip_id"] == trip_id, "train_status"] = TRAIN_STATUS_UNKNOWN
            df_state.loc[df_state["trip_id"] == trip_id, "is_active_now"] = False
            continue

        now_seconds = now_dt.hour * 3600 + now_dt.minute * 60 + now_dt.second
        start_seconds = h1 * 3600 + m1 * 60 + s1
        end_seconds = h2 * 3600 + m2 * 60 + s2

        if now_seconds < start_seconds:
            status = TRAIN_STATUS_UPCOMING
            is_active = False
        elif start_seconds <= now_seconds <= end_seconds:
            status = TRAIN_STATUS_ACTIVE
            is_active = True
        else:
            status = TRAIN_STATUS_FINISHED
            is_active = False

        df_state.loc[df_state["trip_id"] == trip_id, "train_status"] = status
        df_state.loc[df_state["trip_id"] == trip_id, "is_active_now"] = is_active

    return df_state

def load_regularite_history():
    if not HISTORY_FILE.exists():
        return pd.DataFrame()

    return pd.read_excel(HISTORY_FILE)


def get_realtime_df():
    global DF_STATE

    r = requests.get(
        "https://proxy.transport.data.gouv.fr/resource/sncf-gtfs-rt-trip-updates",
        timeout=20
    )
    r.raise_for_status()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(r.content)

    if DF_STATE is None:
        today_str = datetime.today().strftime("%Y%m%d")

        if STATE_FILE.exists():
            DF_STATE = pd.read_excel(STATE_FILE)

            if "date_trip" in DF_STATE.columns:
                file_date = str(DF_STATE["date_trip"].dropna().iloc[0])
            elif "date" in DF_STATE.columns:
                file_date = str(DF_STATE["date"].dropna().iloc[0])
            else:
                file_date = None

            if file_date != today_str:
                DF_STATE = DF_THEORIQUE.copy()
                DF_STATE = enrich_with_voyageurs_descendants(DF_STATE)
        else:
            DF_STATE = DF_THEORIQUE.copy()
            DF_STATE = enrich_with_voyageurs_descendants(DF_STATE)

        if "date_trip" not in DF_STATE.columns:
            DF_STATE["date_trip"] = DF_STATE["date"]

        for col, default in {
            "updated_at": int(time.time()),
            "arrival_delay_s": 0,
            "departure_delay_s": 0,
            "arrival_real": DF_STATE["arrival_time"],
            "departure_real": DF_STATE["departure_time"],
            "retard_s": 0,
            "retard_m": 0.0,
            "train_status": TRAIN_STATUS_UPCOMING,
            "is_active_now": False,
            "is_stop_deleted": False,
            "stop_status": "scheduled",
        }.items():
            if col not in DF_STATE.columns:
                DF_STATE[col] = default
        

    DF_STATE["is_stop_deleted"] = DF_STATE["is_stop_deleted"].fillna(False).astype(bool)
    DF_STATE["stop_status"] = DF_STATE["stop_status"].fillna("scheduled").astype(str)
    DF_STATE = update_df_with_feed(DF_STATE, feed)
    DF_STATE = recompute_train_status(DF_STATE)
    DF_STATE["updated_at"] = int(time.time())

    trains_ok = DF_STATE.loc[
        DF_STATE["stop_name"].isin(LISTE_GARES),
        "train_number"
    ].unique()

    df_result = DF_STATE[DF_STATE["train_number"].isin(trains_ok)].copy()
    try:
        df_alerts = get_service_alerts_by_train()

        df_result["train_number"] = df_result["train_number"].astype(str)
        df_alerts["train_number"] = df_alerts["train_number"].astype(str)

        # IMPORTANT : supprimer les anciennes colonnes vides avant merge
        cols_alertes = ["cause", "titre_alerte", "description_alerte"]
        df_result = df_result.drop(columns=[c for c in cols_alertes if c in df_result.columns])

        df_result = df_result.merge(
            df_alerts[["train_number", "cause", "titre_alerte", "description_alerte"]],
            on="train_number",
            how="left"
        )

        df_result["cause"] = df_result["cause"].fillna("")
        df_result["titre_alerte"] = df_result["titre_alerte"].fillna("")
        df_result["description_alerte"] = df_result["description_alerte"].fillna("")

    except Exception as e:
        debug_log(f"Erreur récupération alertes : {e}")
        df_result["cause"] = ""
        df_result["titre_alerte"] = ""
        df_result["description_alerte"] = ""  

       
    # =========================
    # SUPPRESSION DES TRAINS SANS VOYAGEURS
    # (hors ALEO / Allemagne)
    # =========================

    def normalize_name(x):
        if pd.isna(x):
            return ""
        return str(x).lower().strip()

    gares_allemagne_norm = [
        normalize_name(g)
        for g in LISTE_GARES_ALLEMAGNE
    ]

    def has_gare_allemande(stop_name):
        stop = normalize_name(stop_name)
        return any(
            gare in stop or stop in gare
            for gare in gares_allemagne_norm
        )

    # voyageurs totaux par train
    voyageurs_par_train = (
        df_result
        .groupby("trip_id")["voyageurs_descendants"]
        .sum()
    )

    # présence gare allemande par train
    train_avec_allemagne = (
        df_result
        .groupby("trip_id")["stop_name"]
        .apply(lambda gares: gares.apply(has_gare_allemande).any())
    )

    # trains à supprimer
    trains_a_supprimer = voyageurs_par_train[
        (voyageurs_par_train == 0)
        & (~train_avec_allemagne)
    ].index

    # suppression
    df_result = df_result[
        ~df_result["trip_id"].isin(trains_a_supprimer)
    ].copy()

    debug_log(
        f"Trains supprimés sans voyageurs : {list(trains_a_supprimer)}"
    )  
    save_state_to_excel(df_result)

    return df_result
