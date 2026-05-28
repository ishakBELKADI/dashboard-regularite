from dash import Dash, html, dash_table, dcc, Input, Output, State, ctx, ALL
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from data_loader import (
    get_realtime_df,
    load_regularite_history,
    compute_regularite_metrics,
    save_regularite_history,
    normalize_gare_name,
    LISTE_GARES_ALLEMAGNE
)

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "Dashboard Régularité"

REFRESH_MS = 120000
SEUIL_REGULARITE_S = 359

#La le changement
COLORS_REG = {
    "Global": "#2563eb",
    "RR": "#ef4444",
    "ALEO": "#10b981",
    "IS": "#8b5cf6",
}

COLORS_REGULARITE = {
    "Global": "#2563eb",   # bleu
    "RR": "#ef4444",       # rouge
    "ALEO": "#10b981",     # vert
    "IS": "#8b5cf6",       # violet
}

GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "autoScale2d",
    ],
}

def style_regularity_figure(fig, y_title, legend_title):
    fig.update_traces(
        mode="lines",
        line=dict(width=3, shape="linear"),
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Heure : %{x|%H:%M}<br>"
            "Régularité : %{y:.2f}%"
            "<extra></extra>"
        ),
    )

    for trace in fig.data:
        if trace.name in COLORS_REG:
            trace.line.color = COLORS_REG[trace.name]

    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=55, r=115, t=80, b=50),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        font=dict(family="Arial", size=13, color="#1f2937"),

        title=dict(
            font=dict(size=21, color="#0f172a"),
            x=0.02,
            xanchor="left",
        ),

        xaxis=dict(
            title="Heure",
            dtick=3600000,
            tickformat="%H:%M",
            showgrid=False,
            zeroline=False,
            linecolor="#cbd5e1",
            linewidth=1,
            tickfont=dict(size=12, color="#475569"),
        ),

        yaxis=dict(
            title=y_title,
            range=[50, 100],
            ticksuffix=" %",
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=False,
            linecolor="#cbd5e1",
            linewidth=1,
            tickfont=dict(size=12, color="#475569"),
        ),

        legend=dict(
            title=legend_title,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            x=1.02,
            y=0.98,
            font=dict(size=13),
        ),
    )

    return fig

def gtfs_to_datetime(date_str, time_str):
    if pd.isna(date_str) or pd.isna(time_str) or not date_str or not time_str:
        return pd.NaT
    try:
        date_str = str(date_str)
        h, m, s = map(int, str(time_str).split(":"))
        base = datetime.strptime(date_str, "%Y%m%d")
        return base + timedelta(hours=h, minutes=m, seconds=s)
    except Exception:
        return pd.NaT


def enrich_passage_flags(df):
    df = df.copy()
    now_paris = datetime.now(ZoneInfo("Europe/Paris")).replace(tzinfo=None)

    df["arrival_dt_real"] = df.apply(
        lambda r: gtfs_to_datetime(r.get("date_trip"), r.get("arrival_real")),
        axis=1
    )

    df["desserte_traversee"] = (
        df["arrival_dt_real"].notna() & (df["arrival_dt_real"] <= now_paris)
    )

    return df

def apply_retard_metier_desserte(df):
    df = df.copy()
    df["retard_metier_desserte_s"] = df["retard_s"]

    gares_allemagne_norm = [
        normalize_gare_name(g)
        for g in LISTE_GARES_ALLEMAGNE
    ]

    def is_gare_allemande(stop_name):
        stop = normalize_gare_name(stop_name)
        return any(gare in stop or stop in gare for gare in gares_allemagne_norm)

    for trip_id, train_df in df.groupby("trip_id"):
        train_df = train_df.sort_values("stop_sequence")

        fr_df = train_df[
            ~train_df["stop_name"].apply(is_gare_allemande)
        ].copy()

        if fr_df.empty:
            continue

        incoming_from_germany = is_gare_allemande(train_df.iloc[0]["stop_name"])

        if incoming_from_germany:
            de_df = train_df[
                train_df["stop_name"].apply(is_gare_allemande)
            ].copy()

            derniere_gare_de = de_df.iloc[-1]

            if "departure_delay_s" in de_df.columns:
                retard_reference = derniere_gare_de["departure_delay_s"]
            else:
                retard_reference = derniere_gare_de["retard_s"]

            if pd.isna(retard_reference):
                retard_reference = 0

            for idx, row in fr_df.iterrows():
                df.loc[idx, "retard_metier_desserte_s"] = max(
                    0,
                    row["retard_s"] - retard_reference
                )
            return df


def kpi_card(title, value, subtitle="", bg="white", color="#111"):
    return html.Div(
        style={
            "backgroundColor": bg,
            "color": color,
            "borderRadius": "18px",
            "padding": "18px",
            "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
            "height": "140px",
        },
        children=[
            html.Div(title, style={"fontSize": "18px", "fontWeight": "600", "marginBottom": "10px"}),
            html.Div(value, style={"fontSize": "46px", "fontWeight": "800", "lineHeight": "1"}),
            html.Div(subtitle, style={"fontSize": "15px", "marginTop": "10px", "opacity": "0.85"}),
        ],
    )

def section_title(title, subtitle=""):
    return html.Div(
        style={
            "margin": "8px 0 6px 4px",
            "paddingLeft": "4px",
            "borderLeft": "4px solid #2563eb",
            "lineHeight": "1.2",
        },
        children=[
            html.Div(
                title,
                style={
                    "fontSize": "18px",
                    "fontWeight": "800",
                    "color": "#0f172a",
                },
            ),
            html.Div(
                subtitle,
                style={
                    "fontSize": "12px",
                    "color": "#64748b",
                    "marginTop": "2px",
                },
            ) if subtitle else None,
        ],
    )

def style_regularite_figure(fig, title, y_title):
    fig.update_traces(
        mode="lines",
        line=dict(width=4, shape="spline", smoothing=0.8),
        hovertemplate="<b>%{fullData.name}</b><br>%{x|%H:%M}<br>%{y:.2f}%<extra></extra>",
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b><br><span style='font-size:14px;color:#64748b'>Taux de régularité pendant la journée</span>",
            x=0.02,
            xanchor="left",
            font=dict(size=24, color="#0f172a"),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=430,
        margin=dict(l=70, r=130, t=90, b=60),
        xaxis=dict(
            title="Heure",
            dtick=3600000,
            tickformat="%H:%M",
            showgrid=False,
            zeroline=False,
            linecolor="#cbd5e1",
            tickfont=dict(size=13, color="#475569"),
        ),
        yaxis=dict(
            title=y_title,
            range=[50, 100],
            ticksuffix="%",
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=False,
            linecolor="#cbd5e1",
            tickfont=dict(size=13, color="#475569"),
        ),
        legend=dict(
            title="<b>Régularité</b>",
            bgcolor="white",
            bordercolor="#e5e7eb",
            borderwidth=1,
            x=1.02,
            y=0.95,
            font=dict(size=13),
        ),
        hovermode="x unified",
    )

    return fig


POPUP_STYLE_CLOSED = {
    "display": "none",
    "position": "fixed",
    "top": "0",
    "left": "0",
    "width": "100%",
    "height": "100%",
    "backgroundColor": "rgba(0,0,0,0.45)",
    "zIndex": "9999",
    "justifyContent": "center",
    "alignItems": "center",
}

POPUP_STYLE_OPEN = {
    "display": "flex",
    "position": "fixed",
    "top": "0",
    "left": "0",
    "width": "100%",
    "height": "100%",
    "backgroundColor": "rgba(0,0,0,0.45)",
    "zIndex": "9999",
    "justifyContent": "center",
    "alignItems": "center",
}


app.layout = html.Div(
    style={
        "backgroundColor": "#f3f4f6",
        "minHeight": "100vh",
        "fontFamily": "Arial, sans-serif",
        "margin": "0",
        "padding": "0",
    },
    children=[
        dcc.Interval(id="refresh", interval=REFRESH_MS, n_intervals=0),
        html.Div(
            style={
                "backgroundColor": "black",
                "color": "white",
                "padding": "12px 24px",
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
            },
            children=[
                html.Div("SNCF VOYAGES", style={"fontSize": "28px", "fontWeight": "800"}),
                html.Div("Dashboard Régularité", style={"fontSize": "22px", "fontWeight": "700"}),
                html.Div(
                    [
                        html.Div(id="last-update", style={"fontSize": "16px"}),
                        html.Div(id="debug-refresh", style={"fontSize": "12px", "opacity": "0.7", "marginTop": "2px"})
                    ]
                ),
            ],
        ),
        html.Div(
            id="popup-accostage-surveillance",
            style=POPUP_STYLE_CLOSED,
            children=[
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "width": "85%",
                        "maxHeight": "85%",
                        "overflowY": "auto",
                        "borderRadius": "16px",
                        "padding": "20px",
                        "boxShadow": "0 8px 30px rgba(0,0,0,0.25)",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.H3("Top 20 dessertes restantes à surveiller", style={"margin": "0"}),
                                html.Button("Fermer", id="close-popup-accostage", n_clicks=0),
                            ],
                        ),
                        dash_table.DataTable(
                            id="table-popup-accostage",
                            page_size=20,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "textAlign": "left",
                                "padding": "10px",
                                "fontSize": "13px",
                                "borderBottom": "1px solid #ddd",
                            },
                            style_header={
                                "fontWeight": "bold",
                                "backgroundColor": "#eef1f4",
                            },
                            markdown_options={"html": True},
                        ),
                    ],
                )
            ],
        ),

        html.Div(
            id="popup-gare-regularite",
            style=POPUP_STYLE_CLOSED,
            children=[
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "width": "85%",
                        "maxHeight": "85%",
                        "overflowY": "auto",
                        "borderRadius": "16px",
                        "padding": "20px",
                    },
                    children=[
                        dcc.Store(id="selected-gare-regularite"),

                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between"},
                            children=[
                                html.H3(id="titre-popup-gare-regularite"),
                                html.Button("Fermer", id="close-popup-gare-regularite", n_clicks=0),
                            ],
                        ),

                        dcc.Tabs(
                            id="tabs-gare-regularite",
                            value="regulier",
                            children=[
                                dcc.Tab(label="Trains réguliers", value="regulier"),
                                dcc.Tab(label="Trains irréguliers", value="irregulier"),
                            ],
                        ),

                        dash_table.DataTable(
                            id="table-popup-gare-regularite",
                            page_size=15,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto", "marginTop": "12px"},
                            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
                            style_header={"fontWeight": "bold", "backgroundColor": "#eef1f4"},
                            markdown_options={"html": True},
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="popup-regularite-desserte",
            style=POPUP_STYLE_CLOSED,
            children=[
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "width": "85%",
                        "maxHeight": "85%",
                        "overflowY": "auto",
                        "borderRadius": "16px",
                        "padding": "20px",
                        "boxShadow": "0 8px 30px rgba(0,0,0,0.25)",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.H3("Dernières dessertes arrivées", style={"margin": "0"}),
                                html.Button("Fermer", id="close-popup-regularite-desserte", n_clicks=0),
                            ],
                        ),

                        dcc.Tabs(
                            id="tabs-regularite-desserte",
                            value="regulier",
                            children=[
                                dcc.Tab(label="Réguliers", value="regulier"),
                                dcc.Tab(label="Non réguliers", value="non_regulier"),
                            ],
                        ),

                        dash_table.DataTable(
                            id="table-popup-regularite-desserte",
                            page_size=5,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto", "marginTop": "12px"},
                            style_cell={
                                "textAlign": "left",
                                "padding": "10px",
                                "fontSize": "13px",
                                "borderBottom": "1px solid #ddd",
                            },
                            style_header={
                                "fontWeight": "bold",
                                "backgroundColor": "#eef1f4",
                            },
                        ),
                    ],
                )
            ],
        ),

        html.Div(
            style={"padding": "18px"},
            children=[
                section_title("Vue d’ensemble", "Indicateurs principaux de la journée"),
                html.Div(
                    id="top-kpis",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr 1fr 1fr",
                        "gap": "16px",
                        "marginBottom": "18px",
                    },
                ),

                section_title("Pilotage de la régularité", "Suivi de la régularité à la desserte et au terminus"),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1.2fr 0.8fr",
                        "gap": "18px",
                        "marginBottom": "18px",
                    },
                    children=[
                    html.Div(
                     style={"display": "grid", "gap": "18px"},
                        children=[
                                html.Div(
                                    style={
                                        "backgroundColor": "white",
                                        "borderRadius": "18px",
                                        "padding": "12px",
                                        "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
                                    },
                                    children=[
                                        dcc.RadioItems(
                                            id="choix-regularite-history",
                                            options=[
                                                {"label": "Global", "value": "global"},
                                                {"label": "Radiaux", "value": "rr"},
                                                {"label": "ALEO", "value": "aleo"},
                                                {"label": "IS", "value": "intersecteurs"},
                                                {"label": "Tout", "value": "tout"},
                                            ],
                                            value="global",
                                            inline=True,
                                            style={
                                                "marginBottom": "10px",
                                                "fontWeight": "600",
                                                "display": "flex",
                                                "gap": "18px",
                                            },
                                        ),
                                        dcc.Graph(id="regularite-desserte-history-chart", config=GRAPH_CONFIG)
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "backgroundColor": "white",
                                        "borderRadius": "18px",
                                        "padding": "12px",
                                        "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
                                    },
                                    children=[
                                        dcc.Checklist(
                                                id="checklist-regularite-terminus",
                                                options=[
                                                    {"label": "Global", "value": "global"},
                                                    {"label": "Radiaux", "value": "rr"},
                                                    {"label": "ALEO", "value": "aleo"},
                                                    {"label": "IS", "value": "intersecteurs"},
                                                ],
                                                value=["global"],
                                                inline=True,
                                                style={"marginBottom": "10px"},
                                                labelStyle={
                                                    "display": "inline-block",
                                                    "marginRight": "18px",
                                                    "fontWeight": "bold",
                                                    "fontSize": "18px",
                                                    "cursor": "pointer",
                                                },
                                                inputStyle={
                                                    "marginRight": "6px",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                            html.Div(
                                                id="kpis-terminus-detail",
                                                style={
                                                    "display": "flex",
                                                    "gap": "24px",
                                                    "marginBottom": "8px",
                                                    "fontWeight": "700",
                                                    "fontSize": "16px",
                                                },
                                            ),
                                        dcc.Graph(id="regularite-terminus-history-chart", config=GRAPH_CONFIG),
                                    ]                                ),

                                html.Div(
                                    style={
                                        "backgroundColor": "white",
                                        "borderRadius": "18px",
                                        "padding": "12px",
                                        "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
                                    },
                                    children=[dcc.Graph(id="retard-histogram-chart")],
                                ),

                                html.Div(
                                    style={
                                        "backgroundColor": "white",
                                        "borderRadius": "18px",
                                        "padding": "12px",
                                        "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
                                    },
                                    children=[dcc.Graph(id="regularite-trains-chart")],
                                ),
                            ],
                        ),
                        html.Div(
                            style={"display": "grid", "gap": "18px"},
                            children=[
                                html.Div(id="main-regularite-card"),
                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "1fr 1fr",
                                        "gap": "16px",
                                    },
                                    children=[
                                        html.Div(
                                            id="card-retard-30min",
                                            n_clicks=0,
                                            style={"cursor": "pointer"},
                                        ),
                                        html.Div(
                                            id="card-suppressions",
                                            n_clicks=0,
                                            style={"cursor": "pointer"},
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "backgroundColor": "white",
                                        "borderRadius": "18px",
                                        "padding": "12px",
                                        "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
                                    },
                                    children=[dcc.Graph(id="status-chart")],
                                ),
                                html.Div(
                                    style={
                                        "backgroundColor": "white",
                                        "borderRadius": "18px",
                                        "padding": "12px",
                                        "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
                                    },
                                    children=[
                                        dcc.Input(
                                            id="input-train-voyageurs-regularite",
                                            type="text",
                                            placeholder="Numéro de train",
                                            debounce=True,
                                            style={
                                                "padding": "8px 10px",
                                                "borderRadius": "8px",
                                                "border": "1px solid #ccc",
                                                "width": "180px",
                                                "marginBottom": "10px",
                                            },
                                        ),
                                        dcc.Graph(id="terminus-chart"),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                section_title("Gares les moins régulières", "Top 5 des gares avec le taux de régularité le plus faible"),
                html.Div(
                            id="kpi-gares-moins-regulieres",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(5, 1fr)",
                                "gap": "16px",
                                "marginBottom": "18px",
                            },
                        ),
                section_title("Régularité par gare", "Classement des gares de l’Axe Est selon la régularité à l’arrivée"),

                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "borderRadius": "18px",
                                "padding": "12px",
                                "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
                                "marginBottom": "18px",
                            },
                            children=[
                                dcc.Graph(id="regularite-gares-chart", config=GRAPH_CONFIG)
                            ],
                        ),

                        section_title(
                            "Voyageurs réguliers / irréguliers par gare",
                            "Comparaison du volume voyageurs par gare hors gares allemandes"
                        ),

                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "borderRadius": "18px",
                                "padding": "12px",
                                "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
                                "marginBottom": "18px",
                            },
                            children=[
                                dcc.Graph(id="voyageurs-regularite-gares-chart", config=GRAPH_CONFIG)
                            ],
                        ),


                html.Div(
                    style={
                        "backgroundColor": "white",
                        "borderRadius": "18px",
                        "padding": "12px",
                        "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
                    },
                    
                        children=[
                            html.H3("Détail des trains", style={"marginLeft": "6px"}),

                            dcc.Input(
                                id="search-detail-train",
                                type="text",
                                placeholder="Rechercher un numéro de train",
                                debounce=True,
                                style={
                                    "padding": "8px 10px",
                                    "borderRadius": "8px",
                                    "border": "1px solid #ccc",
                                    "width": "240px",
                                    "margin": "8px 0 14px 6px",
                                },
                            ),

                            dash_table.DataTable(
                            id="table-trains",
                            page_size=20,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto"},
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "#fafafa",
                                }
                                ],
                            style_cell={
                             "textAlign": "left",
                             "padding": "10px",
                             "fontSize": "13px",
                             "borderBottom": "1px solid #ddd",
                             "maxWidth": "220px",
                             "whiteSpace": "normal",
                                },
                            style_header={
                                "fontWeight": "bold",
                                "backgroundColor": "#eef1f4",
                            },
                        ),
                    ],
                ),
            ],
        ),

        html.Div(
            id="popup-retards",
            style=POPUP_STYLE_CLOSED,
            children=[
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "width": "80%",
                        "maxHeight": "80%",
                        "overflowY": "auto",
                        "borderRadius": "16px",
                        "padding": "20px",
                        "boxShadow": "0 8px 30px rgba(0,0,0,0.25)",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "marginBottom": "12px",
                                "gap": "12px",
                            },
                            children=[
                                html.H3("Dessertes en retard > 5 min 59 s", style={"margin": "0"}),
                                html.Div(
                                    style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                    children=[
                                        dcc.Input(
                                            id="search-train-popup",
                                            type="text",
                                            placeholder="Rechercher un numéro de train",
                                            debounce=True,
                                            style={
                                                "padding": "8px 10px",
                                                "borderRadius": "8px",
                                                "border": "1px solid #ccc",
                                                "width": "240px",
                                            },
                                        ),
                                        html.Button("Fermer", id="close-popup", n_clicks=0),
                                    ],
                                ),
                            ],
                        ),
                        dash_table.DataTable(
                            id="popup-table-retards",
                            page_size=15,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
                            style_header={"fontWeight": "bold", "backgroundColor": "#eef1f4"},
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="popup-retards-30min",
            style=POPUP_STYLE_CLOSED,
            children=[
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "width": "80%",
                        "maxHeight": "80%",
                        "overflowY": "auto",
                        "borderRadius": "16px",
                        "padding": "20px",
                        "boxShadow": "0 8px 30px rgba(0,0,0,0.25)",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.H3("Dessertes avec retard > 30 min", style={"margin": "0"}),
                                html.Div(
                                    style={"display": "flex", "alignItems": "center", "gap": "10px"},
                                    children=[
                                        dcc.Input(
                                            id="search-train-popup-30min",
                                            type="text",
                                            placeholder="Rechercher un numéro de train",
                                            debounce=True,
                                            style={
                                                "padding": "8px 10px",
                                                "borderRadius": "8px",
                                                "border": "1px solid #ccc",
                                                "width": "240px",
                                            },
                                        ),
                                        html.Button("Fermer", id="close-popup-30min", n_clicks=0),
                                    ],
                                ),
                            ],
                        ),

                        dash_table.DataTable(
                            id="popup-table-retards-30min",
                            page_size=15,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto"},
                            style_cell={"textAlign": "left", "padding": "8px", "fontSize": "13px"},
                            style_header={"fontWeight": "bold", "backgroundColor": "#eef1f4"},
                            style_data_conditional=[
                            {
                                "if": {"filter_query": "{depart} = ''"},
                                "backgroundColor": "#dc2626",
                                "color": "white",
                                "fontWeight": "bold",
                            }
                        ],
                             markdown_options={"html": True},
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="popup-suppressions",
            style=POPUP_STYLE_CLOSED,
            children=[
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "width": "85%",
                        "maxHeight": "85%",
                        "overflowY": "auto",
                        "borderRadius": "16px",
                        "padding": "20px",
                        "boxShadow": "0 8px 30px rgba(0,0,0,0.25)",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.H3("Arrêts supprimés par train", style={"margin": "0"}),
                                html.Button("Fermer", id="close-popup-suppressions", n_clicks=0),
                            ],
                        ),
                        html.Div(id="content-popup-suppressions"),
                    ],
                )
            ],
        ),
        html.Div(
    id="popup-trains",
    style=POPUP_STYLE_CLOSED,
    children=[
        html.Div(
            style={
                "backgroundColor": "white",
                "width": "85%",
                "maxHeight": "85%",
                "overflowY": "auto",
                "borderRadius": "16px",
                "padding": "20px",
                "boxShadow": "0 8px 30px rgba(0,0,0,0.25)",
            },
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "marginBottom": "12px",
                    },
                    children=[
                        html.H3("Suivi des trains", style={"margin": "0"}),
                        html.Button("Fermer", id="close-popup-trains", n_clicks=0),
                    ],
                ),

                dcc.Input(
                    id="search-train-popup-trains",
                    type="text",
                    placeholder="Rechercher un numéro de train",
                    debounce=True,
                    style={
                        "padding": "8px 10px",
                        "borderRadius": "8px",
                        "border": "1px solid #ccc",
                        "width": "240px",
                        "marginBottom": "12px",
                    },
                ),

                dcc.Tabs(
                    id="tabs-trains",
                    value="encours",
                    children=[
                        dcc.Tab(label="Trains en cours", value="encours"),
                        dcc.Tab(label="Trains en retard", value="retard", id="tab-retard"),
                        dcc.Tab(label="Trains irréguliers", value="irreguliers", id="tab-irreguliers"),       
                        ],
                ),
                html.Div(
                        id="container-subtabs-retard",
                        style={"display": "none"},
                        children=[
                            dcc.Tabs(
                                id="subtabs-retard-trains",
                                value="retard_encours",
                                children=[
                                dcc.Tab(label="En cours", value="retard_encours", id="subtab-retard-encours"),
                                dcc.Tab(label="Terminés", value="retard_finished", id="subtab-retard-finished"),          
                               ],
                            )
                        ]
                ),
                html.Div(
                    id="container-subtabs-irreguliers",
                    style={"display": "none"},
                    children=[
                        dcc.Tabs(
                            id="subtabs-irreguliers-trains",
                            value="irreguliers_encours",
                            children=[
                                dcc.Tab(label="En cours", value="irreguliers_encours", id="subtab-irreguliers-encours"),
                                dcc.Tab(label="Terminés", value="irreguliers_finished", id="subtab-irreguliers-finished"),
                            ],
                        )
                    ],
                ),

                dash_table.DataTable(
                    id="popup-table-trains",
                    page_size=15,
                    sort_action="native",
                    filter_action="native",
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#fafafa",
                        }
                    ],
                    style_header={
                        "fontWeight": "bold",
                        "backgroundColor": "#eef1f4",
                    },
                    style_table={
                        "overflowX": "auto",
                        "maxHeight": "70vh",
                        "overflowY": "auto",
                    },

                    style_cell={
                        "textAlign": "left",
                        "padding": "10px",
                        "fontSize": "13px",
                        "borderBottom": "1px solid #ddd",
                        "whiteSpace": "normal",
                        "height": "auto",
                    },

                    style_cell_conditional=[
                        {"if": {"column_id": "raison"}, "width": "320px", "maxWidth": "320px"},
                        {"if": {"column_id": "train_number"}, "width": "80px"},
                        {"if": {"column_id": "nb_dessertes"}, "width": "90px"},
                        {"if": {"column_id": "retard_moyen_m"}, "width": "120px"},
                        {"if": {"column_id": "retard_max_m"}, "width": "110px"},
                    ],
                ),
            ],
        )
    ],
),
        html.Div(
            id="popup-terminus",
            style=POPUP_STYLE_CLOSED,
            children=[
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "width": "85%",
                        "maxHeight": "85%",
                        "overflowY": "auto",
                        "borderRadius": "16px",
                        "padding": "20px",
                        "boxShadow": "0 8px 30px rgba(0,0,0,0.25)",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "marginBottom": "12px",
                            },
                            children=[
                                html.H3("Trains non réguliers au terminus", style={"margin": "0"}),
                                html.Button("Fermer", id="close-popup-terminus", n_clicks=0),
                            ],
                        ),

                        dcc.Input(
                            id="search-train-popup-terminus",
                            type="text",
                            placeholder="Rechercher un numéro de train",
                            debounce=True,
                            style={
                                "padding": "8px 10px",
                                "borderRadius": "8px",
                                "border": "1px solid #ccc",
                                "width": "240px",
                                "marginBottom": "12px",
                            },
                        ),

                        dash_table.DataTable(
                            id="popup-table-terminus",
                            page_size=15,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto"},
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "#fafafa",
                                }
                            ],
                            style_cell={
                                "textAlign": "left",
                                "padding": "10px",
                                "fontSize": "13px",
                                "borderBottom": "1px solid #ddd",
                            },
                            style_header={
                                "fontWeight": "bold",
                                "backgroundColor": "#eef1f4",
                            },
                        ),
                    ],
                )
            ],
        ),

    ],
)


@app.callback(
    Output("last-update", "children"),
    Output("debug-refresh", "children"),
    Output("top-kpis", "children"),
    Output("main-regularite-card", "children"),
    Output("card-retard-30min", "children"),
    Output("card-suppressions", "children"),
    Output("regularite-trains-chart", "figure"),
    Output("regularite-desserte-history-chart", "figure"),
    Output("regularite-terminus-history-chart", "figure"),
    Output("kpis-terminus-detail", "children"),
    Output("retard-histogram-chart", "figure"),
    Output("status-chart", "figure"),
    Output("terminus-chart", "figure"),
    Output("kpi-gares-moins-regulieres", "children"),
    Output("regularite-gares-chart", "figure"),
    Output("voyageurs-regularite-gares-chart", "figure"),
    Output("table-trains", "data"),
    Output("table-trains", "columns"),
    Input("refresh", "n_intervals"),
    Input("choix-regularite-history", "value"),
    Input("checklist-regularite-terminus", "value"),
    Input("input-train-voyageurs-regularite", "value"),
    Input("search-detail-train", "value"),

)

def update_dashboard(
    n,
    choix_regularite,
    choix_regularite_terminus,
    train_voyageurs_value,
    search_detail_train
):    
    print(f"🔥 update_dashboard déclenché | n={n}", flush=True)

    df = get_realtime_df()
    metrics = compute_regularite_metrics(df)
    save_regularite_history(metrics)
     


    # =========================================================
    # RETARD METIER DESSERTE (LOGIQUE ALEO FRANCE)
    # =========================================================

    df["retard_metier_desserte_s"] = df["retard_s"]

    gares_allemagne_norm = [
        normalize_gare_name(g)
        for g in LISTE_GARES_ALLEMAGNE
    ]

    def is_gare_allemande_app(stop_name):
        stop = normalize_gare_name(stop_name)
        return any(
            gare in stop or stop in gare
            for gare in gares_allemagne_norm
        )

    for trip_id, train_df in df.groupby("trip_id"):

        train_df = train_df.sort_values("stop_sequence")

        # Gares FR uniquement
        fr_df = train_df[
            ~train_df["stop_name"].apply(is_gare_allemande_app)
        ].copy()

        if fr_df.empty:
            continue

        # Train venant d'Allemagne ?
        first_global = train_df.iloc[0]
        incoming_from_germany = is_gare_allemande_app(
            first_global["stop_name"]
        )

        if incoming_from_germany:

            de_df = train_df[
                train_df["stop_name"].apply(is_gare_allemande_app)
            ].copy()

            derniere_gare_de = de_df.iloc[-1]

            if "departure_delay_s" in de_df.columns:
                retard_reference = derniere_gare_de["departure_delay_s"]
            else:
                retard_reference = derniere_gare_de["retard_s"]

            if pd.isna(retard_reference):
                retard_reference = 0

            for idx, row in fr_df.iterrows():

                retard_metier = row["retard_s"] - retard_reference

                if retard_metier < 0:
                    retard_metier = 0

                df.loc[idx, "retard_metier_desserte_s"] = retard_metier
        else:
            # Train France -> Allemagne
            # On garde le retard FR uniquement
            for idx, row in fr_df.iterrows():
                df.loc[idx, "retard_metier_desserte_s"] = row["retard_s"]






    


    # =========================
    # ACCOSTAGE DESSERTE PONDÉRÉ VOYAGEURS
    # =========================

    LISTE_GARES_ALLEMAGNE_APP = [
        "Sarrebruck", "Kaiserslautern Hbf", "Mannheim Hbf",
        "Francfort sur le Main", "Karlsruhe Hbf", "Stuttgart Hbf",
        "Ulm Hbf", "Augsburg Hbf", "Munich", "Offenburg",
        "Lahr (Schwarzw)", "Emmendingen", "Fribourg-en-Brisgau",
        "Baden-Baden",
    ]

    def is_gare_allemande_app(stop_name):
        if pd.isna(stop_name):
            return False
        stop = str(stop_name).lower().strip()
        return any(g.lower().strip() in stop or stop in g.lower().strip() for g in LISTE_GARES_ALLEMAGNE_APP)

    df_acc = df.copy()
    df_acc = enrich_passage_flags(df_acc)

    df_acc = df_acc[~df_acc["stop_name"].apply(is_gare_allemande_app)].copy()

    df_acc["voyageurs_descendants"] = pd.to_numeric(
        df_acc.get("voyageurs_descendants", 0),
        errors="coerce"
    ).fillna(0)

    df_acc["desserte_reguliere"] = (
        df_acc["retard_metier_desserte_s"] <= SEUIL_REGULARITE_S
    )
    df_acc_actuelles = df_acc[
        (df_acc["train_status"] == "finished")
        |
        (
            (df_acc["train_status"] == "active")
            & (df_acc["desserte_traversee"])
        )
    ].copy()

    df_acc_restantes = df_acc.drop(index=df_acc_actuelles.index).copy()

    voyageurs_total_journee = df_acc["voyageurs_descendants"].sum()

    voyageurs_reguliers_actuels = df_acc_actuelles.loc[
        df_acc_actuelles["desserte_reguliere"],
        "voyageurs_descendants"
    ].sum()

    voyageurs_restants = df_acc_restantes["voyageurs_descendants"].sum()

    def taux_accostage(voyageurs_reguliers_proj):
        return (
            round(voyageurs_reguliers_proj / voyageurs_total_journee * 100, 2)
            if voyageurs_total_journee > 0
            else 0
        )

    # Cas idéal : toutes les dessertes restantes sont régulières
    taux_accostage_desserte = taux_accostage(
        voyageurs_reguliers_actuels + voyageurs_restants
    )

    def scenario_x_dessertes_non_regulieres(nb):
        df_pires = (
            df_acc_restantes
            .sort_values("voyageurs_descendants", ascending=False)
            .head(nb)
        )

        voyageurs_non_reguliers_scenario = df_pires["voyageurs_descendants"].sum()

        voyageurs_reguliers_proj = (
            voyageurs_reguliers_actuels
            + voyageurs_restants
            - voyageurs_non_reguliers_scenario
        )

        return taux_accostage(voyageurs_reguliers_proj)

    taux_accostage_desserte_scenario_10 = scenario_x_dessertes_non_regulieres(10)
    taux_accostage_desserte_scenario_20 = scenario_x_dessertes_non_regulieres(20)
    taux_accostage_desserte_scenario_50 = scenario_x_dessertes_non_regulieres(50)

    # Pire état : toutes les dessertes restantes deviennent non régulières
    taux_accostage_desserte_ko = taux_accostage(voyageurs_reguliers_actuels)
    nb_dessertes_restantes = len(df_acc_restantes)
    nb_dessertes_projetees = len(df_acc)


    df = df.copy()
    df["desserte_reguliere"] = df["retard_s"] <= SEUIL_REGULARITE_S
    df["en_retard"] = df["retard_s"] > SEUIL_REGULARITE_S   
    df = enrich_passage_flags(df)
    # Régularité par gare (sur arrivée uniquement)
    # Régularité par gare Axe Est uniquement (sur arrivée uniquement)
    LISTE_GARES_AXE_EST = [
        "Nancy", "Reims", "Charleville-Mézières", "Sedan",
        "Champagne-Ardenne TGV", "Meuse TGV", "Lorraine TGV",
        "Remiremont", "Sarrebourg", "Strasbourg", "Metz",
        "Thionville", "Epinal", "Lunéville", "Saverne",
        "Sélestat", "Colmar", "Mulhouse", "Paris Est", "Luxembourg"
    ]

    df_gare = df[
        (df["desserte_traversee"]) &
        (df["arrival_delay_s"].notna()) &
        (df["stop_name"].isin(LISTE_GARES_AXE_EST))
    ].copy()

    df_gare["arrivee_reguliere"] = (
        df_gare["retard_metier_desserte_s"] <= SEUIL_REGULARITE_S
    )
    df_gare["voyageurs_descendants"] = pd.to_numeric(
    df_gare.get("voyageurs_descendants", 0),
    errors="coerce"
).fillna(0)

    df_gare["voyageurs_reguliers"] = df_gare["voyageurs_descendants"].where(
        df_gare["arrivee_reguliere"],
        0
    )

    regularite_gare = (
        df_gare.groupby("stop_name", as_index=False)
        .agg(
            nb_trains=("trip_id", "count"),
            nb_reguliers=("arrivee_reguliere", "sum"),
            voyageurs_total=("voyageurs_descendants", "sum"),
            voyageurs_reguliers=("voyageurs_reguliers", "sum"),
        )
    )

    regularite_gare["regularite_pct"] = regularite_gare.apply(
        lambda r: round(r["voyageurs_reguliers"] / r["voyageurs_total"] * 100, 2)
        if r["voyageurs_total"] > 0
        else 0,
        axis=1
    )
    regularite_gare = regularite_gare[
        regularite_gare["voyageurs_total"] > 0
    ].copy()

    regularite_gare = regularite_gare.sort_values(
        "regularite_pct",
        ascending=True
    )

    gares_moins_regulieres = (
        regularite_gare
        .sort_values(["regularite_pct", "voyageurs_total"], ascending=[True, False])
        .head(5)
    )

    kpi_gares_moins_regulieres = [
    html.Div(
        id={"type": "card-gare-reg", "index": row["stop_name"]},
        n_clicks=0,
        style={"cursor": "pointer"},
        children=kpi_card(
            row["stop_name"],
            f"{row['regularite_pct']} %",
            f"{int(row['voyageurs_reguliers'])} / {int(row['voyageurs_total'])} voyageurs réguliers"
        )
    )
    for _, row in gares_moins_regulieres.iterrows()
    ]












    if not regularite_gare.empty:
        regularite_gare_chart = regularite_gare.sort_values("regularite_pct", ascending=True)

        fig_regularite_gares = px.bar(
            regularite_gare_chart,
            x="stop_name",
            y="regularite_pct",
            text="regularite_pct",
            title="Régularité par gare - Axe Est",
            labels={
                "stop_name": "Gare",
                "regularite_pct": "Régularité (%)",
            },
        )

        fig_regularite_gares.update_traces(
            texttemplate="%{text:.2f} %",
            textposition="outside",
        )

        fig_regularite_gares.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Gare",
            yaxis_title="Régularité (%)",
            yaxis_range=[0, 100],
            xaxis_tickangle=-35,
            height=460,
        )
    else:
        fig_regularite_gares = px.bar(title="Régularité par gare - aucune donnée")

    df_voy_gare = df[
        (df["desserte_traversee"]) &
        (df["stop_name"].isin(LISTE_GARES_AXE_EST))
    ].copy()

    df_voy_gare["voyageurs_descendants"] = pd.to_numeric(
        df_voy_gare.get("voyageurs_descendants", 0),
        errors="coerce"
    ).fillna(0)

    df_voy_gare["statut_regularite"] = df_voy_gare["retard_metier_desserte_s"].apply(
        lambda x: "Voyageurs réguliers"
        if x <= SEUIL_REGULARITE_S
        else "Voyageurs irréguliers"
    )

    voyageurs_gare_resume = (
        df_voy_gare
        .groupby(["stop_name", "statut_regularite"], as_index=False)
        .agg(voyageurs=("voyageurs_descendants", "sum"))
    )

    voyageurs_gare_resume = voyageurs_gare_resume[
        voyageurs_gare_resume["voyageurs"] > 0
    ].copy()

    if not voyageurs_gare_resume.empty:
        fig_voyageurs_gares = px.bar(
            voyageurs_gare_resume,
            x="stop_name",
            y="voyageurs",
            color="statut_regularite",
            barmode="group",
            text="voyageurs",
            title="Voyageurs réguliers / irréguliers par gare",
            labels={
                "stop_name": "Gare",
                "voyageurs": "Voyageurs descendants",
                "statut_regularite": "Statut",
            },
            color_discrete_map={
                "Voyageurs réguliers": "#16a34a",
                "Voyageurs irréguliers": "#dc2626",
            },
        )

        fig_voyageurs_gares.update_traces(textposition="outside")

        fig_voyageurs_gares.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Gare",
            yaxis_title="Voyageurs descendants",
            xaxis_tickangle=-35,
            height=500,
        )
    else:
        fig_voyageurs_gares = px.bar(
            title="Voyageurs réguliers / irréguliers par gare - aucune donnée"
        )

    trains_uniques = df["trip_id"].nunique() if not df.empty else 0
    
    trains_active = (
        df.groupby("trip_id")["train_status"]
        .first()
        .eq("active")
        .sum()
        if not df.empty else 0
    )

    trains_en_retard = (
        df[df["train_status"].isin(["active", "finished"])]
        .groupby("trip_id")["retard_s"]
        .max()
        .gt(0)
        .sum()
        if not df.empty else 0
    )   
    trains_irreguliers = (
    df[df["train_status"].isin(["active", "finished"])]
    .groupby("trip_id")["retard_s"]
    .max()
    .gt(SEUIL_REGULARITE_S)
    .sum()
    )

    dessertes_retard = int(
        ((df["desserte_traversee"]) & (df["en_retard"])).sum()
    ) if not df.empty else 0
    # Dessertes avec retard > 30 min
    dessertes_retard_30 = (
        df[
            (df["desserte_traversee"]) &
            (df["retard_s"] >= 1800)
        ]
    )

    nb_dessertes_retard_30 = len(dessertes_retard_30)

    nb_trains_retard_30 = (
        dessertes_retard_30["trip_id"].nunique()
        if not dessertes_retard_30.empty
        else 0
    )
    retard_30_card = kpi_card(
        "Dessertes en retard ≥ 30 min",
        str(nb_dessertes_retard_30),
        f"Dont {nb_trains_retard_30} trains concernés",
        bg="white",
        color="#dc2626",
    )
    df_suppressions = df[df.get("is_stop_deleted", False) == True].copy()

    nb_suppressions = len(df_suppressions)

    nb_trains_suppressions = (
        df_suppressions["trip_id"].nunique()
        if not df_suppressions.empty
        else 0
    )

    suppressions_card = kpi_card(
        "Suppressions",
        str(nb_suppressions),
        f"Dont {nb_trains_suppressions} trains concernés",
        bg="white",
        color="#f97316",
    )

    retard_moyen = round(df["retard_m"].mean(), 2) if not df.empty else 0

    top_kpis = [
        html.Div(
                id="card-trains-suivis",
                n_clicks=0,
                style={"cursor": "pointer"},
                children=kpi_card(
                    "Trains suivis",
                    str(trains_uniques),
                    f"{trains_active} en cours | {trains_en_retard} trains en retard | {trains_irreguliers} trains présentant des irrégularités"
                ),
        ),
        html.Div(
            id="card-dessertes-retard",
            n_clicks=0,
            style={"cursor": "pointer"},
            children=kpi_card(
                "Dessertes en retard",
                str(dessertes_retard),
                "Dessertes déjà passées | Seuil > 5 min 59 s"
            ),
        ),
        html.Div(
            id="card-regularite-terminus",
            n_clicks=0,
            style={"cursor": "pointer"},
            children=kpi_card(
                "Régularité terminus",
                f"{metrics['taux_regularite_terminus']} %",
                f"{metrics['nb_trains_terminus_reguliers']} / {metrics['nb_trains_terminus']} trains"
            ),
        ),
        kpi_card(
            "Retard moyen",
            f"{retard_moyen} min",
            "Tous arrêts confondus"
        ),
    ]

    taux_desserte = metrics["taux_regularite_desserte"]
    main_bg = "#7ac000" if taux_desserte >= 90 else "#ffb400" if taux_desserte >= 80 else "#d9043d"

    main_card = html.Div(
        id="card-regularite-desserte",
        n_clicks=0,
        style={
        "cursor": "pointer",            "backgroundColor": "white",
            "borderRadius": "18px",
            "padding": "18px",
            "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
        },
        children=[
            html.Div("Référence métier", style={"fontSize": "20px", "textAlign": "center", "color": "#6b8e99"}),
            html.Div("Régularité à la desserte", style={"fontSize": "22px", "textAlign": "center", "color": "#6b8e99", "marginBottom": "16px"}),
            html.Div(
                f"{taux_desserte} %",
                style={
                    "backgroundColor": main_bg,
                    "color": "white",
                    "fontSize": "92px",
                    "fontWeight": "800",
                    "textAlign": "center",
                    "borderRadius": "18px",
                    "padding": "16px 10px",
                    "lineHeight": "1",
                },
            ),

            html.Div(
                f"Dessertes régulières : {metrics['nb_dessertes_regulieres']} / {metrics['nb_dessertes']}",
                style={"textAlign": "center", "fontSize": "18px", "marginTop": "16px", "color": "#555"},
            ),

            html.Div(
                f"Régularité Radiaux : {metrics['taux_regularite_rr']} %",
                style={
                    "textAlign": "center",
                    "fontSize": "26px",
                    "fontWeight": "700",
                    "marginTop": "18px",
                    "color": "#111",
                },
            ),

            html.Div(
                f"Dessertes Radiaux régulières : {metrics['nb_dessertes_rr_regulieres']} / {metrics['nb_dessertes_rr']}",
                style={
                    "textAlign": "center",
                    "fontSize": "16px",
                    "marginTop": "6px",
                    "color": "#555",
                },
            ),
            html.Div(
            f"Régularité ALEO : {metrics['taux_regularite_aleo']} %",
            style={
                "textAlign": "center",
                "fontSize": "26px",
                "fontWeight": "700",
                "marginTop": "18px",
                "color": "#111",
            },
            ),

            html.Div(
            f"Dessertes ALEO régulières : {metrics['nb_dessertes_aleo_regulieres']} / {metrics['nb_dessertes_aleo']}",
            style={
                "textAlign": "center",
                "fontSize": "16px",
                "marginTop": "6px",
                "color": "#555",
            },
            ),
            html.Div(
                f"Régularité intersecteurs : {metrics['taux_regularite_intersecteurs']} %",
                style={
                    "textAlign": "center",
                    "fontSize": "26px",
                    "fontWeight": "700",
                    "marginTop": "18px",
                    "color": "#111",
                },
            ),

            html.Div(
                f"Dessertes intersecteurs régulières : {metrics['nb_dessertes_intersecteurs_regulieres']} / {metrics['nb_dessertes_intersecteurs']}",
                style={
                    "textAlign": "center",
                    "fontSize": "16px",
                    "marginTop": "6px",
                    "color": "#555",
                },
            ),
        ],
    )
    accostage_card = html.Div(
        id="card-accostage-desserte",
        n_clicks=0,
        style={
            "cursor": "pointer",
            "backgroundColor": "white",
            "borderRadius": "18px",
            "padding": "18px",
            "boxShadow": "0 4px 16px rgba(0,0,0,0.08)",
            "marginTop": "18px",
            "marginBottom": "18px",
            "textAlign": "center",
        },
        children=[
            html.Div(
                "Accostage projeté fin de journée",
                style={"fontSize": "20px", "fontWeight": "700", "color": "#2563eb"},
            ),
            html.Div(
                f"{taux_accostage_desserte} %",
                style={"fontSize": "42px", "fontWeight": "900", "color": "#2563eb", "marginTop": "8px"},
            ),
            html.Div(
                f"Hypothèse : {nb_dessertes_projetees} dessertes restantes régulières",
                style={"fontSize": "15px", "color": "#475569", "marginTop": "6px"},
            ),
            html.Div(
            f"Scénario avec 10 dessertes non régulières : {taux_accostage_desserte_scenario_10} %",
            style={"fontSize": "16px", "fontWeight": "700", "color": "#dc2626", "marginTop": "8px"},
            ),
            html.Div(
            f"Scénario avec 20 dessertes non régulières : {taux_accostage_desserte_scenario_20} %",
            style={"fontSize": "16px", "fontWeight": "700", "color": "#b91c1c", "marginTop": "6px"},
            ),
            html.Div(
            f"Scénario avec 50 dessertes non régulières : {taux_accostage_desserte_scenario_50} %",
            style={"fontSize": "16px", "fontWeight": "700", "color": "#7f1d1d", "marginTop": "6px"},
            ),
            html.Div(
                f"KO pire état possible : {taux_accostage_desserte_ko} %",
                style={"fontSize": "16px", "fontWeight": "800", "color": "#450a0a", "marginTop": "8px"},
            ),

        ],
    )
    



    df_desserte = df[df["train_status"].isin(["active", "finished"])].copy()
    df_desserte = enrich_passage_flags(df_desserte)

    df_desserte = df_desserte[
        (df_desserte["desserte_traversee"])
    ].copy()

    df_desserte["voyageurs_descendants"] = pd.to_numeric(
        df_desserte.get("voyageurs_descendants", 0),
        errors="coerce"
    ).fillna(0)

    def is_gare_allemande_local(stop_name):
        if pd.isna(stop_name):
            return False
        stop = str(stop_name).lower().strip()
        return any(
            g.lower().strip() in stop or stop in g.lower().strip()
            for g in LISTE_GARES_ALLEMAGNE_APP
        )

    df_desserte["is_allemagne"] = df_desserte["stop_name"].apply(is_gare_allemande_local)

    rows = []

    for trip_id, g in df_desserte.sort_values(["trip_id", "stop_sequence"]).groupby("trip_id"):
        g = g.copy()

        # On ne garde que les gares françaises / non allemandes dans le calcul
        g_fr = g[~g["is_allemagne"]].copy()

        if g_fr.empty:
            continue

        first_is_de = bool(g.iloc[0]["is_allemagne"])

        # Allemagne -> France : on neutralise le retard à l'entrée France
        if first_is_de:
            g_de = g[g["is_allemagne"]].copy()
            derniere_gare_de = g_de.iloc[-1]

            if "departure_delay_s" in g_de.columns:
                retard_reference = derniere_gare_de["departure_delay_s"]
            else:
                retard_reference = derniere_gare_de["retard_s"]

            if pd.isna(retard_reference):
                retard_reference = 0

            g_fr["retard_metier_desserte_s"] = (
                g_fr["retard_s"] - retard_reference
            ).clip(lower=0)
        # France -> Allemagne ou train France : retard normal sur périmètre France
        else:
            g_fr["retard_metier_desserte_s"] = g_fr["retard_s"]

        rows.append(g_fr)

    if rows:
        df_desserte = pd.concat(rows, ignore_index=True)
    else:
        df_desserte = pd.DataFrame()

    if not df_desserte.empty:
        df_desserte["desserte_reguliere_train"] = (
            df_desserte["retard_metier_desserte_s"] <= SEUIL_REGULARITE_S
        )

        df_desserte["voyageurs_reguliers"] = df_desserte["voyageurs_descendants"].where(
            df_desserte["desserte_reguliere_train"],
            0
        )

        per_train = (
            df_desserte.groupby("train_number", as_index=False)
            .agg(
                voyageurs_total=("voyageurs_descendants", "sum"),
                voyageurs_reguliers=("voyageurs_reguliers", "sum"),
            )
        )

        per_train = per_train[per_train["voyageurs_total"] > 0].copy()
        per_train["train_number"] = per_train["train_number"].astype(str)

        if not per_train.empty:
            per_train["taux_regularite"] = (
                per_train["voyageurs_reguliers"] / per_train["voyageurs_total"] * 100
            ).round(2)

            per_train = per_train.sort_values("taux_regularite", ascending=True).head(10)
            per_train = per_train.sort_values("taux_regularite", ascending=False)

            fig_regularite = px.bar(
                per_train,
                x="train_number",
                y="taux_regularite",
                title="Top 10 trains les moins réguliers à la desserte",
                text="taux_regularite",
            )
            fig_regularite.update_traces(textposition="outside")
            fig_regularite.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis_title="Train",
                yaxis_title="Régularité voyageurs (%)",
                xaxis={"type": "category"},
            )
        else:
            fig_regularite = px.bar(title="Aucune donnée")
    else:
        fig_regularite = px.bar(title="Aucune donnée")
    if not df_desserte.empty:
        fig_retard = px.histogram(
            df_desserte,
            x="retard_m",
            nbins=30,
            title="Distribution des retards",
        )
        fig_retard.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="Retard (minutes)",
            yaxis_title="Nombre de dessertes",
        )
    else:
        fig_retard = px.histogram(title="Aucune donnée")

    status_df = (
        df[["trip_id", "train_status"]]
        .drop_duplicates()
        .groupby("train_status", as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    if not status_df.empty:
        fig_status = px.pie(
            status_df,
            names="train_status",
            values="count",
            title="Répartition des statuts"
        )
        fig_status.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
    else:
        fig_status = px.pie(title="Aucune donnée")

    df_voy_reg = df.copy()
    df_voy_reg = enrich_passage_flags(df_voy_reg)

    df_voy_reg["voyageurs_descendants"] = pd.to_numeric(
        df_voy_reg.get("voyageurs_descendants", 0),
        errors="coerce"
    ).fillna(0)

    df_voy_reg = df_voy_reg[
        (df_voy_reg["train_status"].isin(["active", "finished"]))
        &
        (df_voy_reg["desserte_traversee"])
    ].copy()

    if train_voyageurs_value:
        df_voy_reg = df_voy_reg[
            df_voy_reg["train_number"].astype(str).str.contains(
                str(train_voyageurs_value),
                case=False,
                na=False
            )
        ].copy()

    if not df_voy_reg.empty:
        df_voy_reg["statut_regularite"] = df_voy_reg["retard_metier_desserte_s"].apply(
            lambda x: "Voyageurs réguliers"
            if x <= SEUIL_REGULARITE_S
            else "Voyageurs non réguliers"
        )
        voy_resume = (
            df_voy_reg.groupby("statut_regularite", as_index=False)
            .agg(voyageurs=("voyageurs_descendants", "sum"))
        )

        fig_terminus = px.bar(
            voy_resume,
            x="statut_regularite",
            y="voyageurs",
            text="voyageurs",
            title="Voyageurs réguliers / non réguliers"
        )

        fig_terminus.update_traces(textposition="outside")

        fig_terminus.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis_title="",
            yaxis_title="Nombre de voyageurs descendants",
            height=360,
        )
    else:
        fig_terminus = px.bar(title="Aucune donnée pour ce train")
    table_cols = [
        "trip_id", "train_number", "stop_name",
        "arrival_time", "departure_time",
        "arrival_real", "departure_real",
        "retard_s", "retard_m", "train_status", "cause"
    ]
    table_df = df[[c for c in table_cols if c in df.columns]].copy()
    if search_detail_train:
        table_df = table_df[
            table_df["train_number"].astype(str).str.contains(
                str(search_detail_train),
                case=False,
                na=False
            )
        ].copy()

    last_update = f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
    debug_refresh = f"Refresh count : {n}"
    df_hist = load_regularite_history()

    if not df_hist.empty:
        df_hist["datetime"] = pd.to_datetime(df_hist["datetime"], errors="coerce")
        df_hist = df_hist.dropna(subset=["datetime"])

        today = datetime.now().date()
        start_dt = datetime.combine(today, datetime.strptime("06:00", "%H:%M").time())
        end_dt = datetime.combine(today, datetime.strptime("22:00", "%H:%M").time())

        df_hist = df_hist[
            (df_hist["datetime"] >= start_dt) &
            (df_hist["datetime"] <= end_dt)
        ].copy()

    if not df_hist.empty:
        colonnes_regularite = {
            "global": ["taux_regularite_desserte"],
            "rr": ["taux_regularite_rr"],
            "aleo": ["taux_regularite_aleo"],
            "intersecteurs": ["taux_regularite_intersecteurs"],
            "tout": [
                "taux_regularite_desserte",
                "taux_regularite_rr",
                "taux_regularite_aleo",
                "taux_regularite_intersecteurs",
            ],
        }

        labels_regularite = {
            "taux_regularite_desserte": "Global",
            "taux_regularite_rr": "RR",
            "taux_regularite_aleo": "ALEO",
            "taux_regularite_intersecteurs": "IS",
        }

        cols = colonnes_regularite.get(
            choix_regularite,
            ["taux_regularite_desserte"]
        )

        fig_history = px.line(
            df_hist,
            x="datetime",
            markers=False,
            y=cols,
            title="Évolution de la régularité à la desserte pendant la journée",
            labels={
                "value": "Régularité (%)",
                "datetime": "Heure",
                "variable": "Type",
            },
        )

        fig_history.for_each_trace(
            lambda t: t.update(
                name=labels_regularite.get(t.name, t.name)
            )
        )
        for trace in fig_history.data:
            trace.line.color = COLORS_REGULARITE.get(trace.name, "#2563eb")

        fig_history = style_regularity_figure(
            fig_history,
            y_title="Régularité (%)",
            legend_title="Régularité",
        )
        colonnes_terminus = {
            "global": "taux_regularite_terminus",
            "rr": "taux_regularite_terminus_rr",
            "aleo": "taux_regularite_terminus_aleo",
            "intersecteurs": "taux_regularite_terminus_intersecteurs",
        }

        labels_terminus = {
            "global": "Global",
            "rr": "RR",
            "aleo": "ALEO",
            "intersecteurs": "IS",
        }

        cols_selected = [
            colonnes_terminus[c]
            for c in choix_regularite_terminus
            if c in colonnes_terminus and colonnes_terminus[c] in df_hist.columns
        ]

        if cols_selected:
            df_plot_terminus = df_hist[["datetime"] + cols_selected].copy()

            df_plot_terminus = df_plot_terminus.rename(
                columns={v: labels_terminus[k] for k, v in colonnes_terminus.items()}
            )

            fig_history_terminus = px.line(
                df_plot_terminus,
                x="datetime",
                y=[labels_terminus[c] for c in choix_regularite_terminus if c in labels_terminus],
                markers=False,
                title="Variation de la régularité au terminus pendant la journée",
            )
        else:
            fig_history_terminus = px.line(
                title="Variation de la régularité au terminus - aucune donnée"
            )

        fig_history_terminus = style_regularity_figure(
            fig_history_terminus,
            y_title="Régularité au terminus (%)",
            legend_title="Type",
        )
    else:
        fig_history = px.line(title="Variation de la régularité à la desserte - aucune donnée")
        fig_history_terminus = px.line(
        title="Variation de la régularité au terminus - aucune donnée"
                )
    
    kpis_terminus_detail = html.Div(
        style={
            "display": "flex",
            "gap": "28px",
            "flexWrap": "wrap",
            "alignItems": "center",
        },
        children=[
            html.Div(f"Global : {metrics['taux_regularite_terminus']} %"),
            html.Div(f"Radiaux : {metrics['taux_regularite_terminus_rr']} %"),
            html.Div(f"ALEO : {metrics['taux_regularite_terminus_aleo']} %"),
            html.Div(f"IS : {metrics['taux_regularite_terminus_intersecteurs']} %"),
        ],
    )

    
    return (
        last_update,
        debug_refresh,
        top_kpis,
        html.Div([main_card, accostage_card]),
        retard_30_card,
        suppressions_card,
        fig_regularite,
        fig_history,
        fig_history_terminus,
        kpis_terminus_detail,
        fig_retard,
        fig_status,
        fig_terminus,
        kpi_gares_moins_regulieres,
        fig_regularite_gares,
        fig_voyageurs_gares,
        table_df.to_dict("records"),
        [{"name": c, "id": c} for c in table_df.columns],
        )


@app.callback(
    Output("popup-retards", "style"),
    Output("popup-table-retards", "data"),
    Output("popup-table-retards", "columns"),
    Input("card-dessertes-retard", "n_clicks"),
    Input("close-popup", "n_clicks"),
    Input("search-train-popup", "value"),
)
def toggle_popup_retards(open_clicks, close_clicks, search_value):
    trigger = ctx.triggered_id

    if trigger == "close-popup":
        return POPUP_STYLE_CLOSED, [], []

    if open_clicks:
        df = get_realtime_df().copy()
        df = enrich_passage_flags(df)
        df = df[df["train_status"].isin(["active", "finished"])].copy()
        df = df[df["desserte_traversee"] & (df["retard_s"] > SEUIL_REGULARITE_S)].copy()
        df = df.sort_values(["trip_id", "stop_sequence"]).copy()
        df["numero_desserte"] = df.groupby("trip_id").cumcount() + 1

        if search_value:
            df = df[df["train_number"].astype(str).str.contains(str(search_value), case=False, na=False)].copy()

        df["type_retard"] = df.apply(
            lambda r: (
                "Arrivée"
                if r["arrival_delay_s"] > r["departure_delay_s"]
                else "Départ"
                if r["departure_delay_s"] > r["arrival_delay_s"]
                else "Arrivée et départ"
            ),
            axis=1,
        )

        def build_train_link(train_number, date_trip):
            train_str = str(train_number).strip()
            if len(train_str) == 5:
                train_str = "0" + train_str
            date_str = str(date_trip) if pd.notna(date_trip) else datetime.now().strftime("%d%m%Y")
            if len(date_str) == 8 and date_str.isdigit():
                date_str = f"{date_str[6:8]}{date_str[4:6]}{date_str[0:4]}"
            return f"https://fichevietrain.sis.sncf.fr/#/detailTrain/{train_str}/date/{date_str}"

        popup_df = df[[
            "train_number",
            "date_trip",
            "numero_desserte",
            "stop_name",
            "voyageurs_descendants",
            "type_retard",
            "retard_s",
            "retard_m",
        ]].copy()
        popup_df["retard_m"] = popup_df["retard_m"].round(2)
        popup_df["voyageurs_descendants"] = (
            pd.to_numeric(popup_df["voyageurs_descendants"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        popup_df["fiche_train"] = popup_df.apply(
            lambda r: f"[Ouvrir]({build_train_link(r['train_number'], r['date_trip'])})",
            axis=1,
        )

        columns = [
            {"name": "Train", "id": "train_number"},
            {"name": "N° desserte", "id": "numero_desserte"},
            {"name": "Gare", "id": "stop_name"},
            {"name": "Voyageurs descendants", "id": "voyageurs_descendants"},
            {"name": "Type retard", "id": "type_retard"},
            {"name": "Retard max (s)", "id": "retard_s"},
            {"name": "Retard max (min)", "id": "retard_m"},
            {"name": "Lien", "id": "fiche_train", "presentation": "markdown"},
        ]
        popup_df = popup_df.drop(columns=["date_trip"])

        return POPUP_STYLE_OPEN, popup_df.to_dict("records"), columns

    return POPUP_STYLE_CLOSED, [], []

@app.callback(
    Output("popup-retards-30min", "style"),
    Output("popup-table-retards-30min", "data"),
    Output("popup-table-retards-30min", "columns"),
    Input("card-retard-30min", "n_clicks"),
    Input("close-popup-30min", "n_clicks"),
    Input("search-train-popup-30min", "value"),
)
def toggle_popup_retards_30min(open_clicks, close_clicks, search_value):
    trigger = ctx.triggered_id

    if trigger == "close-popup-30min":
        return POPUP_STYLE_CLOSED, [], []

    if not open_clicks:
        return POPUP_STYLE_CLOSED, [], []

    df_all = get_realtime_df().copy()
    df_all = enrich_passage_flags(df_all)

    df_retards = df_all[
        (df_all["desserte_traversee"]) &
        (df_all["retard_s"] >= 1800)
    ].copy()

    if search_value:
        df_retards = df_retards[
            df_retards["train_number"].astype(str).str.contains(
                str(search_value), case=False, na=False
            )
        ].copy()

    if df_retards.empty:
        return POPUP_STYLE_OPEN, [], []

    df_retards = df_retards.sort_values(["train_number", "stop_sequence"]).copy()

    def build_train_link(train_number, date_trip):
        train_str = str(train_number).strip()
        if len(train_str) == 5:
            train_str = "0" + train_str

        date_str = str(date_trip) if pd.notna(date_trip) else datetime.now().strftime("%d%m%Y")
        if len(date_str) == 8 and date_str.isdigit():
            date_str = f"{date_str[6:8]}{date_str[4:6]}{date_str[0:4]}"

        return f"https://fichevietrain.sis.sncf.fr/#/detailTrain/{train_str}/date/{date_str}"

    rows = []

    for train_number, g in df_retards.groupby("train_number"):
        train_full = df_all[
            df_all["train_number"].astype(str) == str(train_number)
        ].sort_values("stop_sequence").copy()

        depart = train_full["stop_name"].iloc[0] if not train_full.empty else ""
        terminus = train_full["stop_name"].iloc[-1] if not train_full.empty else ""

        rows.append({
            "train_number": f"🚆 Train {train_number}",
            "depart": "",
            "terminus": "",
            "stop_name": "",
            "voyageurs_descendants": "",
            "retard_m": "",
            "cause": "",
            "fiche_train": "",
        })

        for _, r in g.iterrows():
            cause = str(r.get("cause", "")).strip()
            description = str(r.get("description_alerte", "")).strip()

            cause_complete = (
                f"{cause} - {description}"
                if cause and description and cause != description
                else cause or description
            )

            rows.append({
                "train_number": r["train_number"],
                "depart": depart,
                "terminus": terminus,
                "stop_name": r["stop_name"],
                "voyageurs_descendants": int(r.get("voyageurs_descendants", 0)),
                "retard_m": round(r["retard_m"], 2),
                "cause": cause_complete,
                "fiche_train": f"[Ouvrir]({build_train_link(r['train_number'], r.get('date_trip'))})",
            })

    popup_df = pd.DataFrame(rows)

    columns = [
        {"name": "Train", "id": "train_number"},
        {"name": "Départ", "id": "depart"},
        {"name": "Terminus", "id": "terminus"},
        {"name": "Gare concernée", "id": "stop_name"},
        {"name": "Voyageurs descendants", "id": "voyageurs_descendants"},
        {"name": "Retard (min)", "id": "retard_m"},
        {"name": "Cause", "id": "cause"},
        {"name": "Lien SIS", "id": "fiche_train", "presentation": "markdown"},
    ]

    return POPUP_STYLE_OPEN, popup_df.to_dict("records"), columns

@app.callback(
    Output("popup-trains", "style"),
    Output("popup-table-trains", "data"),
    Output("popup-table-trains", "columns"),
    Input("card-trains-suivis", "n_clicks"),
    Input("close-popup-trains", "n_clicks"),
    Input("tabs-trains", "value"),
    Input("subtabs-retard-trains", "value"),
    Input("search-train-popup-trains", "value"),
    Input("subtabs-irreguliers-trains", "value"),
)
def toggle_popup_trains(open_clicks, close_clicks, tab_value, retard_subtab, search_value, irreguliers_subtab):    
    trigger = ctx.triggered_id

    if trigger == "close-popup-trains":
        return POPUP_STYLE_CLOSED, [], []

    if not open_clicks:
        return POPUP_STYLE_CLOSED, [], []

    df = get_realtime_df().copy()
    df = apply_retard_metier_desserte(df)

    df_scope = df[df["train_status"].isin(["active", "finished"])].copy()
    df_scope = df_scope.sort_values(["trip_id", "stop_sequence"]).copy()

    df_scope = enrich_passage_flags(df_scope)
    df_scope["retard_franche"] = (
        df_scope["desserte_traversee"] &
        (df_scope["retard_s"] > 0)
    )

    df_scope["desserte_prise_en_compte"] = (
        (df_scope["train_status"] == "finished") |
        (
            (df_scope["train_status"] == "active") &
            (df_scope["desserte_traversee"])
        )
    )

    df_scope["desserte_reguliere_calc"] = (
        df_scope["desserte_prise_en_compte"] &
        (df_scope["retard_metier_desserte_s"] <= SEUIL_REGULARITE_S)
    )
    summary = (
        df_scope.groupby("trip_id", as_index=False)
        .agg(
            train_number=("train_number", "first"),
            train_status=("train_status", lambda x: "finished" if (x == "finished").any() else "active" if (x == "active").any() else x.iloc[0]),
            nb_dessertes=("stop_name", "count"),
            gare_depart=("stop_name", "first"),
            terminus=("stop_name", "last"),
            nb_dessertes_en_retard=("retard_franche", "sum"),
            nb_dessertes_irregulieres=("retard_metier_desserte_s", lambda x: int((x > SEUIL_REGULARITE_S).sum())),
            retard_max_brut_s=("retard_s", "max"),
            retard_max_s=("retard_metier_desserte_s", "max"),
            retard_max_m=("retard_metier_desserte_s", lambda x: round(x.max() / 60, 2)),
            retard_moyen_train=("retard_m", lambda x: round(x[x > 0].mean(), 2) if (x > 0).any() else 0),
            date_trip=("date_trip", "first"),
            raison=("cause", lambda x: " | ".join(sorted(set([str(v).strip() for v in x if str(v).strip()])))),
            nb_dessertes_calc=("desserte_prise_en_compte", "sum"),
            nb_dessertes_regulieres_calc=("desserte_reguliere_calc", "sum"),
        )
    )
    summary["taux_regularite_train"] = summary.apply(
    lambda r: round(
        r["nb_dessertes_regulieres_calc"] / r["nb_dessertes_calc"] * 100,
        2
    ) if r["nb_dessertes_calc"] > 0 else 100,
    axis=1
        )
    summary["retard_moyen_train"] = summary["retard_moyen_train"].fillna(0).round(2)
    summary["retard_max_m"] = summary["retard_max_m"].fillna(0).round(2)
    # summary["raison"] = summary.apply(
    #     lambda r: (
    #         f"{r['cause']} - {r['description_alerte']}"
    #         if str(r.get("cause", "")).strip()
    #         and str(r.get("description_alerte", "")).strip()
    #         and str(r.get("cause", "")).strip() != str(r.get("description_alerte", "")).strip()
    #         else str(r.get("cause", "")).strip() or str(r.get("description_alerte", "")).strip()
    #     ),
    #     axis=1
    # )


    if tab_value == "encours":
        summary = summary[summary["train_status"] == "active"].copy()
    elif tab_value == "retard":
        if retard_subtab == "retard_encours":
            summary = summary[
            (summary["train_status"] == "active") &
            (summary["retard_max_brut_s"] > 0)
        ].copy()
        else:
            summary = summary[
            (summary["train_status"] == "finished") &
            (summary["retard_max_brut_s"] > 0)
        ].copy()

    elif tab_value == "irreguliers":
        if irreguliers_subtab == "irreguliers_encours":
            summary = summary[
                (summary["train_status"] == "active") &
                (summary["retard_max_s"] > SEUIL_REGULARITE_S)
            ].copy()
        else:
            summary = summary[
                (summary["train_status"] == "finished") &
                (summary["retard_max_s"] > SEUIL_REGULARITE_S)
            ].copy()
    if search_value:
        summary = summary[
            summary["train_number"].astype(str).str.contains(str(search_value), case=False, na=False)
        ].copy()

    def build_train_link(train_number, date_trip):
        train_str = str(train_number).strip()
        if len(train_str) == 5:
            train_str = "0" + train_str

        date_str = str(date_trip)
        if len(date_str) == 8 and date_str.isdigit():
            date_str = f"{date_str[6:8]}{date_str[4:6]}{date_str[0:4]}"

        return f"https://fichevietrain.sis.sncf.fr/#/detailTrain/{train_str}/date/{date_str}"

    summary["retard_max_m"] = summary["retard_max_m"].round(2)

    summary["fiche_train"] = summary.apply(
        lambda r: f"[Ouvrir]({build_train_link(r['train_number'], r['date_trip'])})",
        axis=1,
    )

    if tab_value == "irreguliers":
        popup_df = summary[[
            "train_number",
            "nb_dessertes",
            "gare_depart",
            "terminus",
            "retard_max_m",
            "taux_regularite_train",
            "fiche_train",
            "raison"
        ]].copy()

        columns = [
            {"name": "Train", "id": "train_number"},
            {"name": "Nb dessertes", "id": "nb_dessertes"},
            {"name": "Gare de départ", "id": "gare_depart"},
            {"name": "Terminus", "id": "terminus"},
            {"name": "Retard max (min)", "id": "retard_max_m"},
            {"name": "Taux de régularité actuel (%)", "id": "taux_regularite_train"},
            {"name": "Cause", "id": "raison"},
            {"name": "Lien SIS", "id": "fiche_train", "presentation": "markdown"},
        ]

    else:
        popup_df = summary[[
            "train_number",
            "nb_dessertes",
            "gare_depart",
            "terminus",
            "nb_dessertes_en_retard",
            "retard_moyen_train",
            "retard_max_m",
            "raison",
            "fiche_train",
        ]].copy()

        columns = [
            {"name": "Train", "id": "train_number"},
            {"name": "Nb dessertes", "id": "nb_dessertes"},
            {"name": "Gare de départ", "id": "gare_depart"},
            {"name": "Terminus", "id": "terminus"},
            {"name": "Nombre de dessertes en retard", "id": "nb_dessertes_en_retard"},
            {"name": "Retard moyen train (min)", "id": "retard_moyen_train"},
            {"name": "Retard max (min)", "id": "retard_max_m"},
            {"name": "Raison", "id": "raison"},
            {"name": "Lien SIS", "id": "fiche_train", "presentation": "markdown"},
        ]
    return POPUP_STYLE_OPEN, popup_df.to_dict("records"), columns


@app.callback(
    Output("container-subtabs-retard", "style"),
    Input("tabs-trains", "value"),
)
def show_hide_subtabs(tab_value):
    if tab_value == "retard":
        return {"display": "block", "marginTop": "8px"}
    return {"display": "none"}




@app.callback(
    Output("container-subtabs-irreguliers", "style"),
    Input("tabs-trains", "value"),
)
def show_hide_subtabs_irreguliers(tab_value):
    if tab_value == "irreguliers":
        return {"display": "block", "marginTop": "8px"}
    return {"display": "none"}

@app.callback(
    Output("tab-retard", "label"),
    Output("subtab-retard-encours", "label"),
    Output("subtab-retard-finished", "label"),
    Output("tab-irreguliers", "label"),
    Output("subtab-irreguliers-encours", "label"),
    Output("subtab-irreguliers-finished", "label"),
    Input("refresh", "n_intervals"),
)
def update_tabs_counts(n):
    df = get_realtime_df().copy()
    df = apply_retard_metier_desserte(df)

    if df.empty:
        return (
            "Trains en retard (0)",
            "En cours (0)",
            "Terminés (0)",
            "Trains irréguliers (0)",
            "En cours (0)",
            "Terminés (0)",
        )

    df = df[df["train_status"].isin(["active", "finished"])].copy()

    summary = (
        df.groupby("trip_id", as_index=False)
        .agg(
            train_status=(
                "train_status",
                lambda x: "finished"
                if (x == "finished").any()
                else "active"
                if (x == "active").any()
                else x.iloc[0],
            ),
        retard_max_brut_s=("retard_s", "max"),
        retard_max_s=("retard_metier_desserte_s", "max"),
        )
    )

    retard_encours = len(summary[
        (summary["train_status"] == "active") &
        (summary["retard_max_brut_s"] > 0)
    ])

    retard_finished = len(summary[
        (summary["train_status"] == "finished") &
        (summary["retard_max_s"] > 0)
    ])

    irreguliers_encours = len(summary[
        (summary["train_status"] == "active") &
        (summary["retard_max_s"] > SEUIL_REGULARITE_S)
    ])

    irreguliers_finished = len(summary[
        (summary["train_status"] == "finished") &
        (summary["retard_max_s"] > SEUIL_REGULARITE_S)
    ])

    total_retard = retard_encours + retard_finished
    total_irreguliers = irreguliers_encours + irreguliers_finished

    return (
        f"Trains en retard ({total_retard})",
        f"En cours ({retard_encours})",
        f"Terminés ({retard_finished})",
        f"Trains irréguliers ({total_irreguliers})",
        f"En cours ({irreguliers_encours})",
        f"Terminés ({irreguliers_finished})",
    )



@app.callback(
    Output("subtabs-irreguliers-trains", "value"),
    Input("tabs-trains", "value"),
)
def reset_subtab_irreguliers(tab_value):
    return "irreguliers_encours"


@app.callback(
    Output("popup-terminus", "style"),
    Output("popup-table-terminus", "data"),
    Output("popup-table-terminus", "columns"),
    Input("card-regularite-terminus", "n_clicks"),
    Input("close-popup-terminus", "n_clicks"),
    Input("search-train-popup-terminus", "value"),
)
def toggle_popup_terminus(open_clicks, close_clicks, search_value):
    trigger = ctx.triggered_id

    if trigger == "close-popup-terminus":
        return POPUP_STYLE_CLOSED, [], []

    if not open_clicks:
        return POPUP_STYLE_CLOSED, [], []

    df = get_realtime_df().copy()
    df_finished = df[df["train_status"] == "finished"].copy()

    if df_finished.empty:
        return POPUP_STYLE_OPEN, [], []

    df_finished = df_finished.sort_values(["trip_id", "stop_sequence"]).copy()

    terminus_df = (
        df_finished.groupby("trip_id", as_index=False)
        .agg(
            train_number=("train_number", "first"),
            nb_dessertes=("stop_name", "count"),
            gare_depart=("stop_name", "first"),
            gare_terminus=("stop_name", "last"),
            voyageurs_descendants=("voyageurs_descendants", "last"),
            retard_terminus_s=("retard_s", "last"),
            retard_terminus_m=("retard_m", "last"),
            date_trip=("date_trip", "first"),
        )
    )
    terminus_df = terminus_df[
        terminus_df["retard_terminus_s"] > SEUIL_REGULARITE_S
    ].copy()

    if search_value:
        terminus_df = terminus_df[
            terminus_df["train_number"].astype(str).str.contains(str(search_value), case=False, na=False)
        ].copy()

    def build_train_link(train_number, date_trip):
        train_str = str(train_number).strip()
        if len(train_str) == 5:
            train_str = "0" + train_str

        date_str = str(date_trip)
        if len(date_str) == 8 and date_str.isdigit():
            date_str = f"{date_str[6:8]}{date_str[4:6]}{date_str[0:4]}"

        return f"https://fichevietrain.sis.sncf.fr/#/detailTrain/{train_str}/date/{date_str}"

    terminus_df["retard_terminus_m"] = terminus_df["retard_terminus_m"].round(2)
    terminus_df["voyageurs_descendants"] = (
        pd.to_numeric(terminus_df["voyageurs_descendants"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    terminus_df["fiche_train"] = terminus_df.apply(
        lambda r: f"[Ouvrir]({build_train_link(r['train_number'], r['date_trip'])})",
        axis=1,
    )

    popup_df = terminus_df[[
        "train_number",
        "nb_dessertes",
        "gare_depart",
        "gare_terminus",
        "retard_terminus_s",
        "retard_terminus_m",
        "voyageurs_descendants",
        "fiche_train",
    ]].copy()

    columns = [
        {"name": "Train", "id": "train_number"},
        {"name": "Nb dessertes", "id": "nb_dessertes"},
        {"name": "Gare de départ", "id": "gare_depart"},
        {"name": "Gare terminus", "id": "gare_terminus"},
        {"name": "Voyageurs descendants", "id": "voyageurs_descendants"},
        {"name": "Retard terminus (s)", "id": "retard_terminus_s"},
        {"name": "Retard terminus (min)", "id": "retard_terminus_m"},
        {"name": "Lien SIS", "id": "fiche_train", "presentation": "markdown"},
    ]

    return POPUP_STYLE_OPEN, popup_df.to_dict("records"), columns


@app.callback(
    Output("popup-suppressions", "style"),
    Output("content-popup-suppressions", "children"),
    Input("card-suppressions", "n_clicks"),
    Input("close-popup-suppressions", "n_clicks"),
)
def toggle_popup_suppressions(open_clicks, close_clicks):
    trigger = ctx.triggered_id

    if trigger == "close-popup-suppressions":
        return POPUP_STYLE_CLOSED, []

    if not open_clicks:
        return POPUP_STYLE_CLOSED, []

    df = get_realtime_df().copy()

    if "is_stop_deleted" not in df.columns:
        return POPUP_STYLE_OPEN, html.Div("Aucune information de suppression disponible.")

    df_sup = df[df["is_stop_deleted"] == True].copy()

    if df_sup.empty:
        return POPUP_STYLE_OPEN, html.Div("Aucune suppression détectée.")

    df_sup = df_sup.sort_values(["train_number", "stop_sequence"]).copy()

    blocs = []

    for train_number, g in df_sup.groupby("train_number"):
        train_full = df[df["train_number"] == train_number].sort_values("stop_sequence").copy()

        depart = train_full["stop_name"].iloc[0] if not train_full.empty else ""
        terminus = train_full["stop_name"].iloc[-1] if not train_full.empty else ""

        table_df = g[["train_number", "stop_name", "stop_sequence"]].copy()

        # Départ / Terminus
        table_df["depart"] = depart
        table_df["terminus"] = terminus

        # Nombre total de dessertes du train
        table_df["nb_dessertes_total"] = len(train_full)

        # Cause (vide pour l’instant)
        # Cause + description alerte
        if "cause" not in g.columns:
            g["cause"] = ""

        if "description_alerte" not in g.columns:
            g["description_alerte"] = ""

        table_df["cause"] = g.apply(
            lambda r: (
                f"{r.get('cause', '')} - {r.get('description_alerte', '')}"
                if str(r.get("cause", "")).strip() and str(r.get("description_alerte", "")).strip()
                else str(r.get("cause", "")).strip() or str(r.get("description_alerte", "")).strip()
            ),
            axis=1
        ).values
        def build_train_link(train_number):
            train_str = str(train_number).strip()
            if len(train_str) == 5:
                train_str = "0" + train_str

            date_str = datetime.now().strftime("%d%m%Y")

            return f"https://fichevietrain.sis.sncf.fr/#/detailTrain/{train_str}/date/{date_str}"
        # Lien SIS
        table_df["fiche_train"] = table_df["train_number"].apply(
            lambda x: f"[Ouvrir]({build_train_link(x)})"
        )

        table_df = table_df.rename(columns={
            "train_number": "Train",
            "depart": "Départ",
            "terminus": "Terminus",
            "nb_dessertes_total": "Nb dessertes total",
            "stop_name": "Desserte supprimée",
            "stop_sequence": "N° desserte",
            "cause": "Cause",
            "fiche_train": "Lien SIS",
        })
        table_df = table_df[[
            "Train",
            "Départ",
            "Terminus",
            "Nb dessertes total",
            "Desserte supprimée",
            "N° desserte",
            "Cause",
            "Lien SIS",
        ]]
        blocs.append(
            html.Div(
                style={
                    "marginBottom": "24px",
                    "border": "1px solid #e5e7eb",
                    "borderRadius": "12px",
                    "overflow": "hidden",
                },
                children=[
                    html.Div(
                        f"Train {train_number}",
                        style={
                            "backgroundColor": "#f97316",
                            "color": "white",
                            "fontWeight": "800",
                            "fontSize": "18px",
                            "padding": "10px 14px",
                        },
                    ),
                    dash_table.DataTable(
                        data=table_df.to_dict("records"),
                        columns=[
                            {
                                "name": c,
                                "id": c,
                                "presentation": "markdown" if c == "Lien SIS" else "input",
                            }
                            for c in table_df.columns
                        ],
                        markdown_options={"html": True},
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "textAlign": "left",
                            "padding": "10px",
                            "fontSize": "13px",
                            "borderBottom": "1px solid #ddd",
                        },
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#fff7ed",
                        },
                    ),
                ],
            )
        )

    return POPUP_STYLE_OPEN, blocs

@app.callback(
    Output("popup-regularite-desserte", "style"),
    Output("table-popup-regularite-desserte", "data"),
    Output("table-popup-regularite-desserte", "columns"),
    Input("card-regularite-desserte", "n_clicks"),
    Input("close-popup-regularite-desserte", "n_clicks"),
    Input("tabs-regularite-desserte", "value"),
)
def toggle_popup_regularite_desserte(open_clicks, close_clicks, tab_value):
    trigger = ctx.triggered_id

    if trigger == "close-popup-regularite-desserte":
        return POPUP_STYLE_CLOSED, [], []

    if not open_clicks:
        return POPUP_STYLE_CLOSED, [], []
    
    gares_allemagne_norm = [
        normalize_gare_name(g)
        for g in LISTE_GARES_ALLEMAGNE
    ]
    
    def is_gare_allemande_popup(stop_name):
        stop = normalize_gare_name(stop_name)
        return any(
            gare in stop or stop in gare
            for gare in gares_allemagne_norm
        )


    df = get_realtime_df().copy()
    df = df[
        ~df["stop_name"].apply(is_gare_allemande_popup)
    ].copy()
    df = enrich_passage_flags(df)

    df = df[
        (df["desserte_traversee"]) &
        (df["arrival_dt_real"].notna())
    ].copy()

    if df.empty:
        return POPUP_STYLE_OPEN, [], []

    df["desserte_reguliere"] = df["retard_s"] <= SEUIL_REGULARITE_S

    df["voyageurs_descendants"] = pd.to_numeric(
        df.get("voyageurs_descendants", 0),
        errors="coerce"
    ).fillna(0).astype(int)

    # Tri du plus récent au plus ancien
    df = df.sort_values("arrival_dt_real", ascending=False)

    # Garde la desserte la plus récente de chaque train
    df_recent = (
        df.drop_duplicates(subset=["trip_id"], keep="first")
        .copy()
    )

    # Re-tri sécurité
    df_recent = df_recent.sort_values(
        "arrival_dt_real",
        ascending=False
    )
    if tab_value == "regulier":
        df_recent = df_recent[df_recent["desserte_reguliere"]].head(5)
    else:
        df_recent = df_recent[~df_recent["desserte_reguliere"]].head(5)

    popup_df = df_recent[[
        "train_number",
        "stop_name",
        "arrival_time",
        "arrival_real",
        "retard_m",
        "voyageurs_descendants",
    ]].copy()

    popup_df["retard_m"] = popup_df["retard_m"].round(2)

    columns = [
        {"name": "Train", "id": "train_number"},
        {"name": "Desserte arrivée", "id": "stop_name"},
        {"name": "Arrivée théorique", "id": "arrival_time"},
        {"name": "Arrivée réelle", "id": "arrival_real"},
        {"name": "Retard arrivée (min)", "id": "retard_m"},
        {"name": "Voyageurs descendants", "id": "voyageurs_descendants"},
    ]

    return POPUP_STYLE_OPEN, popup_df.to_dict("records"), columns

@app.callback(
    Output("popup-accostage-surveillance", "style"),
    Output("table-popup-accostage", "data"),
    Output("table-popup-accostage", "columns"),
    Input("card-accostage-desserte", "n_clicks"),
    Input("close-popup-accostage", "n_clicks"),
)
def toggle_popup_accostage_surveillance(open_clicks, close_clicks):
    trigger = ctx.triggered_id

    if trigger == "close-popup-accostage":
        return POPUP_STYLE_CLOSED, [], []

    if not open_clicks:
        return POPUP_STYLE_CLOSED, [], []

    df = get_realtime_df().copy()
    df = enrich_passage_flags(df)

    LISTE_GARES_ALLEMAGNE_APP = [
        "Sarrebruck", "Kaiserslautern Hbf", "Mannheim Hbf",
        "Francfort sur le Main", "Karlsruhe Hbf", "Stuttgart Hbf",
        "Ulm Hbf", "Augsburg Hbf", "Munich", "Offenburg",
        "Lahr (Schwarzw)", "Emmendingen", "Fribourg-en-Brisgau",
        "Baden-Baden",
    ]

    def is_gare_allemande_app(stop_name):
        if pd.isna(stop_name):
            return False
        stop = str(stop_name).lower().strip()
        return any(
            g.lower().strip() in stop or stop in g.lower().strip()
            for g in LISTE_GARES_ALLEMAGNE_APP
        )

    df = df[~df["stop_name"].apply(is_gare_allemande_app)].copy()

    df["voyageurs_descendants"] = pd.to_numeric(
        df.get("voyageurs_descendants", 0),
        errors="coerce"
    ).fillna(0).astype(int)

    df_restantes = df[
        (df["train_status"] == "upcoming")
        |
        (
            (df["train_status"] == "active")
            & (~df["desserte_traversee"])
        )
    ].copy()

    if df_restantes.empty:
        return POPUP_STYLE_OPEN, [], []

    df_restantes = df_restantes.sort_values(
        "voyageurs_descendants",
        ascending=False
    ).head(20)

    df_restantes["numero_desserte"] = (
        pd.to_numeric(df_restantes["stop_sequence"], errors="coerce")
        .fillna(0)
        .astype(int)
        + 1
    )

    def build_train_link(train_number, date_trip):
        train_str = str(train_number).strip()
        if len(train_str) == 5:
            train_str = "0" + train_str

        date_str = str(date_trip) if pd.notna(date_trip) else datetime.now().strftime("%Y%m%d")
        if len(date_str) == 8 and date_str.isdigit():
            date_str = f"{date_str[6:8]}{date_str[4:6]}{date_str[0:4]}"

        return f"https://fichevietrain.sis.sncf.fr/#/detailTrain/{train_str}/date/{date_str}"

    popup_df = df_restantes[[
        "train_number",
        "numero_desserte",
        "stop_name",
        "arrival_time",
        "departure_time",
        "voyageurs_descendants",
        "date_trip",
    ]].copy()

    popup_df["fiche_train"] = popup_df.apply(
        lambda r: f"[Ouvrir]({build_train_link(r['train_number'], r['date_trip'])})",
        axis=1,
    )

    popup_df = popup_df.drop(columns=["date_trip"])

    columns = [
        {"name": "Train", "id": "train_number"},
        {"name": "N° desserte", "id": "numero_desserte"},
        {"name": "Desserte", "id": "stop_name"},
        {"name": "Arrivée théorique", "id": "arrival_time"},
        {"name": "Départ théorique", "id": "departure_time"},
        {"name": "Voyageurs descendants", "id": "voyageurs_descendants"},
        {"name": "Lien SIS", "id": "fiche_train", "presentation": "markdown"},
    ]

    return POPUP_STYLE_OPEN, popup_df.to_dict("records"), columns



@app.callback(
    Output("popup-gare-regularite", "style"),
    Output("titre-popup-gare-regularite", "children"),
    Output("table-popup-gare-regularite", "data"),
    Output("table-popup-gare-regularite", "columns"),
    Output("selected-gare-regularite", "data"),
    Input({"type": "card-gare-reg", "index": ALL}, "n_clicks"),
    Input("close-popup-gare-regularite", "n_clicks"),
    Input("tabs-gare-regularite", "value"),
    State("selected-gare-regularite", "data"),
)
def toggle_popup_gare_regularite(clicks_gares, close_clicks, tab_value, selected_gare):
    trigger = ctx.triggered_id
    trigger_info = ctx.triggered[0] if ctx.triggered else {}

    if trigger == "close-popup-gare-regularite":
        return POPUP_STYLE_CLOSED, "", [], [], None

    if isinstance(trigger, dict) and trigger.get("type") == "card-gare-reg":
        if not trigger_info.get("value"):
            return POPUP_STYLE_CLOSED, "", [], [], None

        selected_gare = trigger.get("index")

    elif trigger == "tabs-gare-regularite":
        if not selected_gare:
            return POPUP_STYLE_CLOSED, "", [], [], None

    else:
        return POPUP_STYLE_CLOSED, "", [], [], None
    



    df = get_realtime_df().copy()
    df = apply_retard_metier_desserte(df)
    df = enrich_passage_flags(df)

    df = df[
        (df["desserte_traversee"]) &
        (df["stop_name"] == selected_gare)
    ].copy()

    df["voyageurs_descendants"] = pd.to_numeric(
        df.get("voyageurs_descendants", 0),
        errors="coerce"
    ).fillna(0).astype(int)

    df["gare_reguliere"] = df["retard_metier_desserte_s"] <= SEUIL_REGULARITE_S

    if tab_value == "regulier":
        df = df[df["gare_reguliere"]].copy()
    else:
        df = df[~df["gare_reguliere"]].copy()

    def build_train_link(train_number, date_trip):
        train_str = str(train_number).strip()
        if len(train_str) == 5:
            train_str = "0" + train_str

        date_str = str(date_trip) if pd.notna(date_trip) else datetime.now().strftime("%d%m%Y")
        if len(date_str) == 8 and date_str.isdigit():
            date_str = f"{date_str[6:8]}{date_str[4:6]}{date_str[0:4]}"

        return f"https://fichevietrain.sis.sncf.fr/#/detailTrain/{train_str}/date/{date_str}"

    if not df.empty:
        df["retard_metier_m"] = (df["retard_metier_desserte_s"] / 60).round(2)
        df["fiche_train"] = df.apply(
            lambda r: f"[Ouvrir]({build_train_link(r['train_number'], r.get('date_trip'))})",
            axis=1
        )

        popup_df = df[[
            "train_number",
            "stop_name",
            "arrival_time",
            "arrival_real",
            "retard_metier_m",
            "voyageurs_descendants",
            "fiche_train",
        ]].copy()
    else:
        popup_df = pd.DataFrame(columns=[
            "train_number",
            "stop_name",
            "arrival_time",
            "arrival_real",
            "retard_metier_m",
            "voyageurs_descendants",
            "fiche_train",
        ])

    columns = [
        {"name": "Train", "id": "train_number"},
        {"name": "Gare", "id": "stop_name"},
        {"name": "Arrivée théorique", "id": "arrival_time"},
        {"name": "Arrivée réelle", "id": "arrival_real"},
        {"name": "Retard métier (min)", "id": "retard_metier_m"},
        {"name": "Voyageurs descendants", "id": "voyageurs_descendants"},
        {"name": "Lien SIS", "id": "fiche_train", "presentation": "markdown"},
    ]

    return (
        POPUP_STYLE_OPEN,
        f"Trains passés par {selected_gare}",
        popup_df.to_dict("records"),
        columns,
        selected_gare,
    )



if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)