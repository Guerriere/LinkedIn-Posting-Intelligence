"""
LinkedIn Posting Intelligence — v2 (données réelles)
=====================================================
Lancer avec : streamlit run linkedin_dashboard_v2.py

Colonnes attendues dans le CSV :
  username, date, hour, day_of_week, total_reactions, likes, comments,
  reposts, has_image, has_video, post_type, text_length, has_hashtags,
  nb_hashtags, has_emoji, has_link, post_url, text, categorie

Installe les dépendances :
  pip install streamlit pandas plotly scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

JOURS_ORDER = ["lundi","mardi","mercredi","jeudi","vendredi","samedi","dimanche"]
PALETTE = ["#042C53","#185FA5","#378ADD","#85B7EB","#B5D4F4","#1D9E75","#BA7517","#D85A30","#888780"]

# ─── Chargement & enrichissement ───────────────────────────────────────────────
@st.cache_data
def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["year"]  = df["date"].dt.year
    df["day_of_week"] = df["day_of_week"].str.lower().str.strip()
    df["categorie"]   = df["categorie"].str.strip()
    # Nettoyage catégories
    cat_map = {"storytelling.": "storytelling", "annoncé": "annonce", "annonces": "annonce", "opération": "annonce"}
    df["categorie"] = df["categorie"].replace(cat_map)
    df = cap_outliers(df)
    return df
    
def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    cols = ['total_reactions', 'likes', 'comments', 'reposts']
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        borne_sup = Q3 + 1.5 * IQR
        borne_inf = Q1 - 1.5 * IQR
        df[col] = df[col].clip(lower=borne_inf, upper=borne_sup)
    return df

# ─── Sidebar & filtres ─────────────────────────────────────────────────────────
def sidebar(df: pd.DataFrame):
    st.sidebar.header("Filtres")
    uploaded = st.sidebar.file_uploader("Changer de fichier CSV", type=["csv"])
    if uploaded:
        df = load_uploaded(uploaded)
        st.sidebar.success(f"{len(df):,} lignes chargées ✅")

    authors = ["Tous"] + sorted(
    df["username"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)
    sel_author = st.sidebar.selectbox("Créateur", authors)

    cats = ["Toutes"] + sorted(df["categorie"].dropna().unique().tolist())
    sel_cat = st.sidebar.selectbox("Catégorie", cats)

    years = sorted(df["year"].dropna().unique().tolist())
    sel_years = st.sidebar.multiselect("Années", years, default=years[-3:] if len(years) >= 3 else years)

    types_ = ["Tous"] + df["post_type"].unique().tolist()
    sel_type = st.sidebar.selectbox("Type de post", types_)

    metric = st.sidebar.radio("Métrique principale", ["total_reactions","likes","comments","reposts"],
                               format_func=lambda x: x.replace("_"," ").capitalize())
    return df, sel_author, sel_cat, sel_years, sel_type, metric


@st.cache_data
def load_uploaded(f) -> pd.DataFrame:
    return load(f)


def apply_filters(df, sel_author, sel_cat, sel_years, sel_type):
    mask = df["year"].isin(sel_years)
    if sel_author != "Tous":
        mask &= df["username"] == sel_author
    if sel_cat != "Toutes":
        mask &= df["categorie"] == sel_cat
    if sel_type != "Tous":
        mask &= df["post_type"] == sel_type
    return df[mask]


# ─── KPI Row ───────────────────────────────────────────────────────────────────
def kpi_row(df: pd.DataFrame, metric: str):
    best_day   = df.groupby("day_of_week")[metric].mean().idxmax()
    best_hour  = int(df.groupby("hour")[metric].mean().idxmax())
    best_cat   = df.groupby("categorie")[metric].mean().idxmax() if "categorie" in df else "—"
    avg        = df[metric].mean()
    combo = (df.groupby(["day_of_week","hour"])[metric]
               .mean()
               .reset_index()
               .sort_values(metric, ascending=False)
               .iloc[0])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Posts analysés",   f"{len(df):,}")
    c2.metric("Meilleur jour",    best_day.capitalize())
    c3.metric("Meilleure heure",  f"{best_hour}h")
    c4.metric("Top catégorie",    best_cat.capitalize())
    c5.metric(f"{metric.replace('_',' ').capitalize()} moy.", f"{avg:.0f}")
    st.caption(f"Combo gagnant : **{combo['day_of_week'].capitalize()} {int(combo['hour'])}h** "
               f"→ {combo[metric]:.0f} {metric.replace('_',' ')} en moy.")


# ─── Charts ────────────────────────────────────────────────────────────────────
def fig_by_day(df, metric):
    order = [j for j in JOURS_ORDER if j in df["day_of_week"].unique()]
    agg = df.groupby("day_of_week")[metric].mean().reindex(order).reset_index()
    fig = px.bar(agg, x="day_of_week", y=metric,
                 color=metric, color_continuous_scale=["#B5D4F4","#042C53"],
                 labels={"day_of_week":"", metric: metric.replace("_"," ").capitalize()},
                 title="Réactions moyennes par jour")
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=36,b=0,l=0,r=0))
    fig.update_traces(marker_line_width=0)
    return fig


def fig_by_hour(df, metric):
    agg = df.groupby("hour")[metric].mean().reset_index()
    fig = px.area(agg, x="hour", y=metric, line_shape="spline",
                  color_discrete_sequence=["#1D9E75"],
                  labels={"hour":"Heure", metric: metric.replace("_"," ").capitalize()},
                  title="Réactions moyennes par heure")
    fig.update_traces(fill="tozeroy", fillcolor="rgba(29,158,117,0.12)")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=36,b=0,l=0,r=0))
    return fig


def fig_heatmap(df, metric):
    dff = df[df["hour"].between(7, 20)]
    agg = dff.groupby(["day_of_week","hour"])[metric].mean().reset_index()
    pivot = agg.pivot(index="day_of_week", columns="hour", values=metric).fillna(0)
    order = [j for j in JOURS_ORDER if j in pivot.index]
    pivot = pivot.reindex(order)
    fig = px.imshow(pivot, color_continuous_scale=["#E6F1FB","#378ADD","#042C53"],
                    labels={"x":"Heure","y":"Jour","color": metric.replace("_"," ").capitalize()},
                    title="Heatmap : jour × heure (7h–20h)", aspect="auto")
    fig.update_layout(margin=dict(t=36,b=0,l=0,r=0))
    return fig


def fig_by_cat(df, metric):
    agg = (df.groupby("categorie")[metric].mean()
             .reset_index()
             .sort_values(metric, ascending=True))
    fig = px.bar(agg, x=metric, y="categorie", orientation="h",
                 color=metric, color_continuous_scale=["#B5D4F4","#042C53"],
                 labels={"categorie":"", metric: metric.replace("_"," ").capitalize()},
                 title="Réactions par catégorie de contenu")
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=36,b=0,l=0,r=0))
    return fig


def fig_format_impact(df, metric):
    rows = []
    for col, labels in [("has_image",["Sans image","Avec image"]),
                        ("has_video",["Sans vidéo","Avec vidéo"]),
                        ("has_emoji",["Sans emoji","Avec emoji"]),
                        ("has_hashtags",["Sans hashtag","Avec hashtag"]),
                        ("has_link",["Sans lien","Avec lien"])]:
        g = df.groupby(col)[metric].mean()
        delta = ((g.get(1, 0) - g.get(0, 0)) / g.get(0, 1)) * 100 if g.get(0, 0) else 0
        rows.append({"Feature": labels[1], "Avec": round(g.get(1, 0), 1),
                     "Sans": round(g.get(0, 0), 1), "Delta%": round(delta, 1)})
    agg = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_bar(name="Sans", x=agg["Feature"], y=agg["Sans"], marker_color="#B5D4F4")
    fig.add_bar(name="Avec", x=agg["Feature"], y=agg["Avec"], marker_color="#042C53")
    fig.update_layout(barmode="group", title="Impact des éléments de format",
                      plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=36,b=0,l=0,r=0),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig, agg


def fig_text_length(df, metric):
    bins = [0, 100, 300, 600, 1000, 3000]
    labels = ["0–100","100–300","300–600","600–1k","1k–3k"]
    df2 = df.copy()
    df2["len_bin"] = pd.cut(df2["text_length"], bins=bins, labels=labels)
    agg = df2.groupby("len_bin", observed=True)[metric].mean().reset_index()
    fig = px.bar(agg, x="len_bin", y=metric,
                 color=metric, color_continuous_scale=["#B5D4F4","#042C53"],
                 labels={"len_bin":"Longueur du texte", metric: "Moy. réactions"},
                 title="Réactions selon la longueur du texte")
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=36,b=0,l=0,r=0))
    return fig


def fig_top_authors(df, metric, n=10):
    agg = (df.groupby("username")[metric]
             .mean()
             .reset_index()
             .merge(df.groupby("username").size().rename("count").reset_index())
             .query("count >= 10")
             .nlargest(n, metric))
    agg["username"] = agg["username"].str.replace(r"%[0-9A-Fa-f]{2}", "", regex=True).str[:25]
    fig = px.bar(agg, x=metric, y="username", orientation="h",
                 color=metric, color_continuous_scale=["#B5D4F4","#042C53"],
                 labels={"username":"", metric: "Moy. réactions"},
                 title=f"Top {n} créateurs (min. 10 posts)")
    fig.update_layout(coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=36,b=0,l=0,r=0))
    return fig


def fig_post_type(df, metric):
    agg = df.groupby("post_type")[metric].mean().reset_index().sort_values(metric, ascending=False)
    fig = px.bar(agg, x="post_type", y=metric,
                 color="post_type", color_discrete_sequence=["#042C53","#378ADD","#888780"],
                 labels={"post_type":"", metric: "Moy. réactions"},
                 title="Réactions par type de post")
    fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(t=36,b=0,l=0,r=0))
    return fig


def fig_trend(df, metric):
    agg = df.groupby("month")[metric].mean().reset_index()
    fig = px.line(agg, x="month", y=metric, markers=True,
                  color_discrete_sequence=["#185FA5"],
                  labels={"month":"Mois", metric: "Moy. réactions"},
                  title="Tendance mensuelle")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=36,b=0,l=0,r=0))
    return fig


# ─── Prédiction ML ─────────────────────────────────────────────────────────────
@st.cache_resource
def train(df: pd.DataFrame, metric: str):
    df2 = df[["hour","day_of_week","has_image","has_video","has_emoji",
              "has_hashtags","has_link","text_length","categorie", metric]].dropna()
    le_day = LabelEncoder()
    le_cat = LabelEncoder()
    df2 = df2.copy()
    df2["dow_enc"] = le_day.fit_transform(df2["day_of_week"])
    df2["cat_enc"] = le_cat.fit_transform(df2["categorie"])
    feats = ["hour","dow_enc","has_image","has_video","has_emoji",
             "has_hashtags","has_link","text_length","cat_enc"]
    X = df2[feats].values
    y = df2[metric].values
    model = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=42)
    model.fit(X, y)
    return model, le_day, le_cat, feats


def prediction_tab(df, metric):
    st.subheader("Prédiction quel est le meilleur moment pour poster ?")
    model, le_day, le_cat, feats = train(df, metric)

    cats_avail = sorted(df["categorie"].dropna().unique().tolist())
    c1, c2, c3 = st.columns(3)
    sel_day   = c1.selectbox("Jour", [j.capitalize() for j in JOURS_ORDER], index=1)
    sel_hour  = c2.slider("Heure", 6, 22, 8)
    sel_cat   = c3.selectbox("Catégorie", cats_avail, index=cats_avail.index("tip_pratique") if "tip_pratique" in cats_avail else 0)

    c4, c5, c6, c7, c8 = st.columns(5)
    has_image    = int(c4.checkbox("Image", value=True))
    has_video    = int(c5.checkbox("Vidéo"))
    has_emoji    = int(c6.checkbox("Emoji", value=True))
    has_hashtag  = int(c7.checkbox("Hashtag", value=True))
    has_link     = int(c8.checkbox("Lien"))
    text_len     = st.slider("Longueur du texte (caractères)", 50, 3000, 800, step=50)

    dow_enc = le_day.transform([sel_day.lower()])[0] if sel_day.lower() in le_day.classes_ else 0
    cat_enc = le_cat.transform([sel_cat])[0] if sel_cat in le_cat.classes_ else 0
    x = np.array([[sel_hour, dow_enc, has_image, has_video, has_emoji, has_hashtag, has_link, text_len, cat_enc]])
    pred = float(model.predict(x)[0])
    global_mean = df[metric].mean()
    delta_pct = (pred - global_mean) / global_mean * 100

    col_a, col_b = st.columns([2,1])
    col_a.metric(f"Réactions prédites ({metric.replace('_',' ')})",
                 f"{pred:.0f}", f"{delta_pct:+.0f}% vs moyenne globale ({global_mean:.0f})")

    score = min(100, max(0, int((pred / df[metric].quantile(0.9)) * 100)))
    color = "#1D9E75" if score >= 70 else "#BA7517" if score >= 45 else "#A32D2D"
    col_b.markdown(f"""
    <div style='text-align:center;padding:12px;background:#f0f4f8;border-radius:12px;'>
      <div style='font-size:11px;color:#555;'>Score</div>
      <div style='font-size:28px;font-weight:600;color:{color};'>{score}/100</div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.subheader("Calendrier optimal de la semaine")
    rows = []
    for dow in JOURS_ORDER:
        if dow not in le_day.classes_: continue
        best_score, best_h, best_c = -1, 8, cats_avail[0]
        for h in range(6, 22):
            for cat in cats_avail:
                de = le_day.transform([dow])[0]
                ce = le_cat.transform([cat])[0]
                xp = np.array([[h, de, 1, 0, 1, 1, 0, 800, ce]])
                s = float(model.predict(xp)[0])
                if s > best_score:
                    best_score, best_h, best_c = s, h, cat
        rows.append({"Jour": dow.capitalize(), "Heure": f"{best_h}h",
                     "Catégorie": best_c, "Réactions prédites": f"{best_score:.0f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.title("LinkedIn Posting Intelligence")
    st.caption("Basé sur tes vraies données scrappées · 1 573 posts · 64 créateurs")

    df_raw = load("linkedln_post.csv")
    df, sel_author, sel_cat, sel_years, sel_type, metric = sidebar(df_raw)
    df_f = apply_filters(df, sel_author, sel_cat, sel_years, sel_type)

    if df_f.empty:
        st.warning("Aucune donnée avec ces filtres."); return

    kpi_row(df_f, metric)
    st.divider()

    t1, t2, t3, t4, t5 = st.tabs(["Timing", "Format & éléments", "Catégories", "Créateurs", "Prédiction ML"])

    with t1:
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_by_day(df_f, metric),  use_container_width=True)
        c2.plotly_chart(fig_by_hour(df_f, metric), use_container_width=True)
        st.plotly_chart(fig_heatmap(df_f, metric),  use_container_width=True)
        st.plotly_chart(fig_trend(df_f, metric),    use_container_width=True)

    with t2:
        fig_fmt, df_fmt = fig_format_impact(df_f, metric)
        st.plotly_chart(fig_fmt, use_container_width=True)
        st.dataframe(df_fmt, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_text_length(df_f, metric), use_container_width=True)
        c2.plotly_chart(fig_post_type(df_f, metric),   use_container_width=True)
        with st.expander("Voir les posts bruts"):
            cols = ["username","date","hour","day_of_week","total_reactions",
                    "categorie","post_type","has_image","has_video","text_length","text"]
            st.dataframe(df_f[cols].sort_values("total_reactions", ascending=False).head(100),
                         use_container_width=True)

    with t3:
        st.plotly_chart(fig_by_cat(df_f, metric), use_container_width=True)
        cat_stats = (df_f.groupby("categorie")[metric]
                       .agg(["mean","median","count"])
                       .round(1)
                       .sort_values("mean", ascending=False)
                       .rename(columns={"mean":"Moy.","median":"Médiane","count":"Posts"}))
        st.dataframe(cat_stats, use_container_width=True)

    with t4:
        st.plotly_chart(fig_top_authors(df_f, metric), use_container_width=True)
        st.caption(f"{df_f['username'].nunique()} créateurs dans la sélection.")

    with t5:
        prediction_tab(df, metric)


if __name__ == "__main__":
    main()