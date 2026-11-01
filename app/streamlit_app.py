"""Dashboard « Forecast accuracy »."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from hotelforecast import backtest as bt
from hotelforecast.metrics import bias, wape
from hotelforecast.simulate import load_world

BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#0b0b0b"
st.set_page_config(page_title="HotelForecast", layout="wide")


@st.cache_data
def load_results() -> pd.DataFrame:
    return pd.read_parquet(Path("data/outputs/backtest.parquet"))


@st.cache_resource
def load_group():
    return load_world(Path("data/synthetic"))


results, world = load_results(), load_group()
st.title("Précision des prévisions")
st.caption("Groupe hôtelier fictif. Backtest à origine glissante : aucune donnée réelle.")

left, middle, right = st.columns(3)
horizon = left.slider("Horizon maximum (jours)", 7, 90, 30)
brands = middle.multiselect("Marque", sorted(results["brand"].unique()))
regions = right.multiselect("Région", sorted(results["region"].unique()))

view = results[results["horizon"] <= horizon]
if brands:
    view = view[view["brand"].isin(brands)]
if regions:
    view = view[view["region"].isin(regions)]

model_wape = wape(view["actual_rooms"], view["forecast_rooms"])
rms_wape = wape(view["actual_rooms"], view["rms"])
naive_wape = wape(view["actual_rooms"], view["naive_y1"])
a, b, c, d = st.columns(4)
a.metric("WAPE LightGBM", f"{model_wape:.1%}")
b.metric("WAPE RMS simulé", f"{rms_wape:.1%}", f"{rms_wape - model_wape:+.1%} d'écart",
         delta_color="off")
c.metric("WAPE naïve Y-1", f"{naive_wape:.1%}")
d.metric("Biais LightGBM", f"{bias(view['actual_rooms'], view['forecast_rooms']):+.1%}")

curve_tab, heatmap_tab, pickup_tab, drift_tab = st.tabs(
    ["Réalisé / prévu / RMS", "Carte d'erreur", "Montée en charge", "Hôtels en dérive"]
)

with curve_tab:
    hotel = st.selectbox("Hôtel", sorted(results["hotel_id"].unique()))
    fold = st.selectbox("Origine", sorted(results["origin_date"].unique()), index=5,
                        format_func=lambda d: f"{pd.Timestamp(d):%d/%m/%Y}")
    one = results[(results["hotel_id"] == hotel) & (results["origin_date"] == fold)]
    one = one.sort_values("stay_date")
    figure = go.Figure()
    figure.add_scatter(x=one["stay_date"], y=one["p90_rooms"], line_width=0, showlegend=False)
    figure.add_scatter(x=one["stay_date"], y=one["p10_rooms"], line_width=0, fill="tonexty",
                       fillcolor="rgba(42,120,214,0.14)", name="Intervalle à 80 %")
    figure.add_scatter(x=one["stay_date"], y=one["actual_rooms"], name="Réalisé",
                       line={"color": INK, "width": 2})
    figure.add_scatter(x=one["stay_date"], y=one["forecast_rooms"], name="LightGBM",
                       line={"color": BLUE, "width": 2})
    figure.add_scatter(x=one["stay_date"], y=one["rms"], name="RMS simulé",
                       line={"color": ORANGE, "width": 2})
    figure.update_layout(hovermode="x unified", yaxis_title="Chambres vendues", height=440)
    st.plotly_chart(figure, width="stretch")

with heatmap_tab:
    table = bt.error_heatmap(view, horizon)
    table = table.reindex(columns=range(7))
    table.columns = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    st.plotly_chart(
        px.imshow(table * 100, text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
                  labels={"x": "Jour", "y": "Mois", "color": "WAPE (%)"}),
        width="stretch",
    )

with pickup_tab:
    hotel_index = st.selectbox("Hôtel ", range(len(world.hotels)),
                               format_func=lambda i: world.hotels["hotel_id"].iloc[i])
    final = world.rooms_sold[hotel_index, -365:].sum()
    curve = world.otb[hotel_index, -365:, :91].sum(axis=0) / final
    pickup = pd.DataFrame({"Jours avant le séjour": range(91), "Part du final": curve})
    chart = px.line(pickup, x="Jours avant le séjour", y="Part du final")
    chart.update_xaxes(autorange="reversed")
    chart.update_traces(line={"color": BLUE, "width": 2})
    chart.update_yaxes(tickformat=".0%")
    st.plotly_chart(chart, width="stretch")
    st.write({f"J-{d}": f"{curve[d]:.0%}" for d in (30, 14, 7)})

with drift_tab:
    st.write("Hôtels classés par évolution de leur demande réalisée contre Y-1, sur 28 jours.")
    report = bt.drift_report(results, world).head(12)
    st.dataframe(
        report.rename(columns={"is_drifting": "dérive injectée"}).style.format(
            {"bias_28d": "{:+.1%}", "y1_change_28d": "{:+.1%}"}
        ),
        hide_index=True,
    )
