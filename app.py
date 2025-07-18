# app.py
import json
import pandas as pd
import streamlit as st

# ─── Page config (must be first) ───
st.set_page_config(page_title="Institution Profiles Explorer", layout="wide")

@st.cache_data
def load_data():
    with open("profiles.json", "r", encoding="utf-8") as f:
        return pd.json_normalize(json.load(f))

df = load_data()

st.title("🔍 Institution Profiles Explorer")

# ─── Sidebar filters ───
with st.sidebar:
    st.header("Filters")

    # Type dropdown from JSON
    all_types = ["<any>"] + sorted(df["type"].dropna().unique().tolist())
    sel_type = st.selectbox("Type", all_types, index=0)

    # Country dropdown from JSON
    all_countries = ["<any>"] + sorted(df["country"].dropna().unique().tolist())
    sel_country = st.selectbox("Country", all_countries, index=0)

    # Year Founded spinners (numeric inputs)
    years = df["year_founded"].dropna().astype(int)
    if not years.empty:
        min_year, max_year = int(years.min()), int(years.max())
        sel_year_min = st.number_input(
            "Year Founded (min)",
            min_value=min_year,
            max_value=max_year,
            value=min_year,
            step=1
        )
        sel_year_max = st.number_input(
            "Year Founded (max)",
            min_value=min_year,
            max_value=max_year,
            value=max_year,
            step=1
        )
    else:
        sel_year_min = sel_year_max = None

    # Full-text search box
    query = st.text_input("Search (name/desc)")


# ─── Apply filters ───
filtered = df.copy()

# Filter by type
if sel_type != "<any>":
    filtered = filtered[filtered["type"] == sel_type]

# Filter by country
if sel_country != "<any>":
    filtered = filtered[filtered["country"] == sel_country]

# Filter by year, but keep rows where year_founded is null
if sel_year_min is not None:
    in_range = filtered["year_founded"].between(sel_year_min, sel_year_max)
    filtered = filtered[in_range | filtered["year_founded"].isna()]

# Full-text search
if query:
    mask = (
        filtered["name"].str.contains(query, case=False, na=False)
        | filtered["short_description"].str.contains(query, case=False, na=False)
        | filtered["full_profile"].str.contains(query, case=False, na=False)
    )
    filtered = filtered[mask]

# Show match count
st.markdown(f"### {len(filtered)} records matched (out of {len(df)})")

# ─── Results table ───
cols = ["name", "type", "country", "year_founded", "website"]
st.dataframe(
    filtered[cols]
      .rename(columns={
          "name": "Name",
          "type": "Type",
          "country": "Country",
          "year_founded": "Year Founded",
          "website": "Website"
      })
      .reset_index(drop=True),
    height=400,
)

# ─── Detail viewer ───
st.markdown("---")
st.subheader("Detail Viewer")

if not filtered.empty:
    idx = st.number_input(
        "Select record index",
        min_value=0,
        max_value=len(filtered) - 1,
        value=0,
        step=1
    )
    rec = filtered.iloc[int(idx)]
    st.markdown(f"## {rec['name']}")
    st.markdown(
        f"**Type:** {rec['type'] or 'N/A'}  \n"
        f"**Country:** {rec['country'] or 'N/A'}  \n"
        f"**Year Founded:** {rec['year_founded'] or 'N/A'}"
    )
    if pd.notna(rec.get("website")):
        st.markdown(f"[🌐 Website]({rec['website']})")
    st.markdown("### Short Description")
    st.write(rec["short_description"] or "_None_")
    st.markdown("### Full Profile")
    st.write(rec["full_profile"] or "_None_")
else:
    st.warning("No records to display. Adjust your filters.")
