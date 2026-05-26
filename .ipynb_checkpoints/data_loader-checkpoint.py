import pandas as pd
import requests
from datetime import datetime
from zoneinfo import ZoneInfo
from google.transit import gtfs_realtime_pb2
from pathlib import Path

import pandas as pd
import time
from datetime import datetime



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
    
DF_THEORIQUE = load_theoretical_data()

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

    df["desserte_reguliere"] = df["retard_s"] <= SEUIL

    trains_rr = (
        df.groupby("trip_id")["stop_name"]
        .apply(lambda gares: gares.isin(LISTE_GARES).all())
    )

    trip_ids_rr = trains_rr[trains_rr].index
    df_rr = df[df["trip_id"].isin(trip_ids_rr)].copy()

    df_desserte = df[
        (df["train_status"] == TRAIN_STATUS_FINISHED)
        |
        (
            (df["train_status"] == TRAIN_STATUS_ACTIVE)
            & (df["desserte_traversee"])
        )
    ].copy()

    nb_dessertes = len(df_desserte)
    nb_dessertes_regulieres = int(df_desserte["desserte_reguliere"].sum())

    taux_regularite_desserte = (
        nb_dessertes_regulieres / nb_dessertes * 100
        if nb_dessertes > 0 else 0
    )

    df_rr_desserte = df_rr[
        (df_rr["train_status"] == TRAIN_STATUS_FINISHED)
        |
        (
            (df_rr["train_status"] == TRAIN_STATUS_ACTIVE)
            & (df_rr["desserte_traversee"])
        )
    ].copy()

    nb_dessertes_rr = len(df_rr_desserte)
    nb_dessertes_rr_regulieres = int(df_rr_desserte["desserte_reguliere"].sum())

    taux_regularite_rr = (
        nb_dessertes_rr_regulieres / nb_dessertes_rr * 100
        if nb_dessertes_rr > 0 else 0
    )
    # Identifier les trains ALEO :
    # un train est ALEO s'il contient au moins une gare allemande
    def normalize_name(x):
        if pd.isna(x):
            return ""
        return str(x).lower().strip()

    gares_allemagne_norm = [normalize_name(g) for g in LISTE_GARES_ALLEMAGNE]

    def is_gare_allemande(stop_name):
        stop = normalize_name(stop_name)
        return any(gare in stop or stop in gare for gare in gares_allemagne_norm)

    trains_aleo = (
        df.groupby("trip_id")["stop_name"]
        .apply(lambda gares: gares.apply(is_gare_allemande).any())
    )

    trip_ids_aleo = trains_aleo[trains_aleo].index
    df_aleo = df[df["trip_id"].isin(trip_ids_aleo)].copy()

    # REGULARITE A LA DESSERTE - ALEO
    df_aleo_desserte = df_aleo[
        (df_aleo["train_status"] == TRAIN_STATUS_FINISHED)
        |
        (
            (df_aleo["train_status"] == TRAIN_STATUS_ACTIVE)
            & (df_aleo["desserte_traversee"])
        )
    ].copy()

    nb_dessertes_aleo = len(df_aleo_desserte)

    nb_dessertes_aleo_regulieres = int(
        df_aleo_desserte["desserte_reguliere"].sum()
    )

    taux_regularite_aleo = (
        nb_dessertes_aleo_regulieres / nb_dessertes_aleo * 100
        if nb_dessertes_aleo > 0 else 0
    )
    trip_ids_rr = set(trip_ids_rr)
    trip_ids_aleo = set(trip_ids_aleo)

    trip_ids_intersecteurs = set(df["trip_id"].unique()) - trip_ids_rr - trip_ids_aleo

    df_intersecteurs = df[df["trip_id"].isin(trip_ids_intersecteurs)].copy()

    df_intersecteurs_desserte = df_intersecteurs[
        (df_intersecteurs["train_status"] == TRAIN_STATUS_FINISHED)
        |
        (
            (df_intersecteurs["train_status"] == TRAIN_STATUS_ACTIVE)
            & (df_intersecteurs["desserte_traversee"])
        )
    ].copy()

    nb_dessertes_intersecteurs = len(df_intersecteurs_desserte)

    nb_dessertes_intersecteurs_regulieres = int(
        df_intersecteurs_desserte["desserte_reguliere"].sum()
    )

    taux_regularite_intersecteurs = (
        nb_dessertes_intersecteurs_regulieres / nb_dessertes_intersecteurs * 100
        if nb_dessertes_intersecteurs > 0 else 0
    )




    df_finished = df[df["train_status"] == TRAIN_STATUS_FINISHED].copy()

    terminus_df = (
        df_finished.sort_values(["train_number", "stop_sequence"])
        .groupby("train_number", as_index=False)
        .last()
    )

    terminus_df["terminus_regulier"] = terminus_df["retard_s"] <= SEUIL

    nb_trains_terminus = len(terminus_df)
    nb_trains_terminus_reguliers = int(terminus_df["terminus_regulier"].sum())

    taux_regularite_terminus = (
        nb_trains_terminus_reguliers / nb_trains_terminus * 100
        if nb_trains_terminus > 0 else 0
    )
    def compute_terminus_categorie(df_cat):
        df_finished_cat = df_cat[df_cat["train_status"] == TRAIN_STATUS_FINISHED].copy()

        if df_finished_cat.empty:
            return 0, 0, 100

        terminus_cat = (
            df_finished_cat.sort_values(["train_number", "stop_sequence"])
            .groupby("train_number", as_index=False)
            .last()
        )

        terminus_cat["terminus_regulier"] = terminus_cat["retard_s"] <= SEUIL

        nb_trains = len(terminus_cat)
        nb_reguliers = int(terminus_cat["terminus_regulier"].sum())
        taux = nb_reguliers / nb_trains * 100 if nb_trains > 0 else 0

        return nb_trains, nb_reguliers, round(taux, 2)

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
            df_state.at[idx, "retard_s"] = max(arr_delay_s, dep_delay_s)
            df_state.at[idx, "retard_m"] = max(arr_delay_s, dep_delay_s) / 60

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
        else:
            DF_STATE = DF_THEORIQUE.copy()

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

    save_state_to_excel(df_result)

    return df_result
