#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 19:51:06 2025

@author: seansteele
"""

import pandas as pd
import numpy as np
from fredapi import Fred
import eia as Eia 
import requests
from functools import reduce
import time
import datetime
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#%% Functions
def core_eia_apicall(eia_api_key, url):
    res = requests.get(url, params={'api_key': eia_api_key})
    response = res.json()
    response = res.json()['response']
    data = pd.DataFrame(response['data'])
    return data

def paginated_eia_retrival(eia_api_key, url):
    offset = 0
    url_string = url
    live_url = eval(f"f'''{url_string}'''")
    data = []
    res = requests.get(live_url, params={'api_key': eia_api_key})
    response = res.json()['response']
    max_count = response['total']
    while offset < int(max_count):
        retrieved_data = core_eia_apicall(eia_api_key, live_url)
        if len(data) == 0:
            data = retrieved_data.copy()
        else:
            data = pd.concat([data, retrieved_data], ignore_index=True)
        offset = len(data)
        url_string = url
        live_url = eval(f"f'''{url_string}'''")
    return data


def pivot_eia_data(data, source_df, groupby_column, secondary_name, value_name):    
    
    data = data[['period', value_name, groupby_column]]
    data = data.replace([None, 'None'], [np.nan, np.nan]) 
    data.dropna(inplace = True)
    
    data[value_name] = data[value_name].astype(float)
    data.columns = ['date', value_name, groupby_column]
    
    data = data.pivot_table(
        index=['date'],
        columns=groupby_column,
        values=value_name
    ).reset_index()
    
    non_area_cols = ['date']
    padd_cols = [col for col in data.columns if col not in non_area_cols]
    
    data.rename(
        columns={col: col.replace(' ', '').lower() + secondary_name for col in padd_cols},
        inplace=True
    )
    
    return data


#%% Core data Loading
gid = '1838518553'
sheet_id = '1ECugqAU75hiwblVyL-ZUpeS6__qXT90PiRVItSxNuF8'
url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}'
source_df = pd.read_csv(url)

fred_api_key = '557dbdc0b7bf3f0141d86447373fe7d6'
fred = Fred(api_key = fred_api_key)

eia_api_key = 'BbkkVrZpSYqvO5S254ciOd6Ft2aac1cwc9i9e0DC'

fred_datas = []
eia_datas = []
for i in range(len(source_df)):
    if source_df['source'].iloc[i] == 'fred':
        fred_data_name = source_df['data_name'].iloc[i]
        data = pd.DataFrame(fred.get_series(fred_data_name))
        data['date'] = list(data.index.strftime('%Y-%m-%d'))
        data_name = source_df['secondary_name'].iloc[i]
        data.columns = [data_name, 'date']
        fred_datas.append(data)
    if source_df['source'].iloc[i] == 'eia_api':
        url = source_df['data_link'].iloc[i]
        if source_df['paginated'].iloc[i] == 'yes':
            paginated = True
        else:
            paginated = False
            
        if paginated == True:
            data = paginated_eia_retrival(eia_api_key, url)
        else:
            data = core_eia_apicall(eia_api_key, url)
            
        if source_df['column_to_groupby'].isna().iloc[i]:
            data = data[['period', 'value']]
            data.columns = ['date', source_df['secondary_name'].iloc[i]]
        else:
            groupby_column = source_df['column_to_groupby'].iloc[i]
            secondary_name = source_df['secondary_name'].iloc[i]
            value_name = source_df['value_name'].iloc[i]
            data = pivot_eia_data(data, source_df, groupby_column, secondary_name, value_name)
            
            if secondary_name !='_gasprice':
                cols_to_sum = [col for col in data.columns if 'date' not in col.lower()]
                data[f'total{secondary_name}'] = data[cols_to_sum].sum(axis=1)
            
        eia_datas.append(data)
    del data
        

#drop incorrect summation and clean us_gasprice name
    
    
#%% Monthly data aggregation

fred_monthly = reduce(
    lambda left, right: pd.merge(left, right, on="date", how="outer"),
    [x for x in fred_datas]
)
fred_monthly['date'] = pd.to_datetime(fred_monthly['date'])
fred_monthly.set_index('date', inplace=True)
fred_monthly = fred_monthly.resample('ME').mean()
    
eia_monthly = reduce(
    lambda left, right: pd.merge(left, right, on="date", how="outer"),
    [x for x in eia_datas]
)
eia_monthly['us_gasprice'] = eia_monthly['u.s._gasprice']
eia_monthly = eia_monthly.drop(['u.s._gasprice'], axis = 1)
eia_monthly['date'] = pd.to_datetime(eia_monthly['date'], format='mixed')
eia_monthly.set_index('date', inplace=True)
eia_monthly = eia_monthly.astype(float)
eia_monthly = eia_monthly.resample('ME').mean()

all_monthly = pd.merge(fred_monthly, eia_monthly, on="date", how="outer")
all_monthly['inflation_rate'] = 100 * (all_monthly['cpi_all'] - all_monthly['cpi_all'].shift(12))/all_monthly['cpi_all'].shift(12)


#%% Correlation of most montly data

all_corr = all_monthly.corr()

corrs = ['cpi_all', 'inflation_rate', 'all_construction_spend', 'housing_under_construction',
         'egg_price', 'total_manufact_orders', 'population', 'federal_outlays','federal_employees',
         'gas_price', 'oil_price', 'industrial_prod', 'electricity_price', 'all_imports', 'all_exports',
         'us_natural_gas_prod', 'total_crude_oil_prod', 'total_crude_oil_export','total_crude_oil_import',
         'total_million_tons_co2']

corr_df = all_monthly[corrs]
corr_df.columns = ['cpi', 'inflation', 'construction spend', 'housing construction',
         'egg price', 'manufacturing orders', 'population', 'federal spending', 'govt employees',
         'gas price', 'oil price', 'industrial prod', 'electricity price', 'imports', 'exports',
         'natgas prod', 'oil prod', 'oil export','oil import','co2 emission']
corr_df = corr_df.corr()
corr_df = corr_df.round(2)

ax = sns.heatmap(corr_df, annot=corr_df, fmt='.2f', annot_kws={"size": 5.25},
            cmap="coolwarm"
            )
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=6)  
plt.title('Pearson Correlation Between Core Monthly Data')


#recent correlations (2016 onwards)
recent_corr_df = all_monthly[all_monthly.index >= '2016-01-01']
recent_corr_df = recent_corr_df[corrs]
recent_corr_df.columns = ['cpi', 'inflation', 'construction spend', 'housing construction',
         'egg price', 'manufacturing orders', 'population', 'federal spending', 'govt employees',
         'gas price', 'oil price', 'industrial prod', 'electricity price', 'imports', 'exports',
         'natgas prod', 'oil prod', 'oil export','oil import','co2 emission']
recent_corr_df = recent_corr_df.corr()

ax = sns.heatmap(recent_corr_df, annot=recent_corr_df, fmt='.2f', annot_kws={"size": 5.25},
            cmap="coolwarm"
            )
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=6)  
plt.title('Pearson Correlation Between Core Monthly Data')


#%% Federal Employees vs federal outlays

df = all_monthly[all_monthly.index >= '2012-01-01']

fig, ax1 = plt.subplots(figsize=(10, 6))

sns.lineplot(data=df, x="date", y="federal_outlays", ax=ax1, color="blue", label="Federal Spending")
ax1.set_ylabel("Federal Spending", fontsize=12, color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

# Create secondary axis (right)
ax2 = ax1.twinx()
sns.lineplot(data=df, x="date", y="federal_employees", ax=ax2, color="orange", label="Federal Employees")
ax2.set_ylabel("Federal Employees", fontsize=12, color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

# Titles and layout


lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=10)

#%% Crude Oil Prodiction

### Oil production and employee line plots ###
df = all_monthly[all_monthly.index >= '2012-01-01']

series_cols = ['total_crude_oil_prod', 'padd1_crude_oil_prod', 'padd2_crude_oil_prod',
               'padd3_crude_oil_prod', 'padd4_crude_oil_prod', 'padd5_crude_oil_prod']

series_titles = ['Total Crude Oil Production', 'East Coast Production', 'Midwest Production',
                 'Gulf Coast Production', 'Rocky Mountains Production', 'West Coast Production']


fig = plt.figure(figsize=(16, 20))  # total figure size
gs = gridspec.GridSpec(nrows=4, ncols=3, height_ratios=[2, 1.5, 1.5, 1.5], hspace=0.3)

# First plot spans all columns in first row (larger)
ax0 = fig.add_subplot(gs[0, :])
sns.lineplot(data=df, x="date", y=series_cols[0], ax=ax0, color = 'blue')
ax0.set_title(series_titles[0] + '\n', fontsize=22)
ax0.set_xlabel("")
ax0.set_ylabel("Oil Production MMBL/D")

#rest of plots
for i in range(1, 6):
    row = (i + 2) // 3  # maps 1–5 to row indices 1–3
    col = (i - 1) % 3
    ax = fig.add_subplot(gs[row, col])
    sns.lineplot(data=df, x="date", y=series_cols[i], ax=ax)
    ax.set_title(series_titles[i], fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlabel("")
    if col == 0:
        ax.set_ylabel("Oil Production MMBL/D")
    else:
        ax.set_ylabel("")
       
#use empty space to plot employee counts
row = 2
col = 2
ax = fig.add_subplot(gs[row, col])
sns.lineplot(data=df, x="date", y='oil_employees', ax=ax, color = 'purple')
ax.set_title('Total Oil Employees', fontsize=12)
ax.tick_params(labelsize=10)
ax.set_xlabel("")
ax.set_ylabel("Employee Count (Thousands)")

plt.tight_layout()
plt.show()


### plot  percent change from a year ago ###
df = all_monthly.copy()
df['oil_perc_change'] = 100 * (df['total_crude_oil_prod'] - df['total_crude_oil_prod'].shift(12))/df['total_crude_oil_prod'].shift(12)
df = df.iloc[-37:]
df['date'] = df.index
date_labels = list(df['date'])
date_labels = [d.strftime('%m-%Y') for d in date_labels]
df = df[['date', 'oil_perc_change']]
df.dropna(inplace = True)

x = np.array(range(len(df)))
y = df.oil_perc_change.values 
points = np.array([x, y]).T  
segments = np.stack([points[:-1], points[1:]], axis=1)  # shape (N, 2, 2)

# Create LineCollection
norm = plt.Normalize(-6, y.max())
lc = LineCollection(segments, cmap='RdYlGn', norm=norm)
lc.set_array((y[:-1] + y[1:]) / 2)  # value for color at segment midpoints
lc.set_linewidth(2)


fig, ax = plt.subplots(figsize=(10, 4))
ax.add_collection(lc)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
plt.colorbar(lc, ax=ax, label='Percent Change')
plt.ylabel('Percent Change')
ax.set_title("Total Oil Production Percent Change From 1 Year Ago")
tick_locs = np.linspace(0, 36, 10, dtype=int) 
ax.set_xticks(tick_locs)
ax.set_xticklabels([date_labels[i] for i in tick_locs], rotation=90, ha='right')
plt.show()



### plot oil imports and exports ###
### Oil imports/exports line plots ###
df = all_monthly[all_monthly.index >= '2012-01-01']

series_cols = ['total_crude_oil_export', 'padd1_crude_oil_export', 'padd2_crude_oil_export',
               'padd3_crude_oil_export', 'padd4_crude_oil_export', 'padd5_crude_oil_export']

series_titles = ['Total Crude Oil Import', 'East Coast Import', 'Midwest Import',
                 'Gulf Coast Import', 'Rocky Mountains Import', 'West Coast Import']


fig = plt.figure(figsize=(16, 20))  # total figure size
gs = gridspec.GridSpec(nrows=4, ncols=3, height_ratios=[2, 1.5, 1.5, 1.5], hspace=0.3)

# First plot spans all columns in first row (larger)
ax0 = fig.add_subplot(gs[0, :])
sns.lineplot(data=df, x="date", y=series_cols[0], ax=ax0, color = 'blue')
ax0.set_title(series_titles[0] + '\n', fontsize=22)
ax0.set_xlabel("")
ax0.set_ylabel("Oil Import MMBL/D")

#rest of plots
for i in range(1, 6):
    row = (i + 2) // 3  # maps 1–5 to row indices 1–3
    col = (i - 1) % 3
    ax = fig.add_subplot(gs[row, col])
    sns.lineplot(data=df, x="date", y=series_cols[i], ax=ax)
    ax.set_title(series_titles[i], fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_xlabel("")
    if col == 0:
        ax.set_ylabel("Oil Import MMBL/D")
    else:
        ax.set_ylabel("")
       
#use empty space to plot employee counts
row = 2
col = 2
ax = fig.add_subplot(gs[row, col])
sns.lineplot(data=df, x="date", y='oil_employees', ax=ax, color = 'purple')
ax.set_title('Total Oil Employees', fontsize=12)
ax.tick_params(labelsize=10)
ax.set_xlabel("")
ax.set_ylabel("Employee Count (Thousands)")

plt.tight_layout()
plt.show()

#%% gasoline dashboard


app = dash.Dash(__name__)

# Your input data
gas_df = all_monthly[['padd1a_gasprice', 'padd1b_gasprice', 'padd1c_gasprice',
                      'padd2_gasprice', 'padd3_gasprice', 'padd4_gasprice',
                      'padd5_gasprice', 'us_gasprice']].dropna()
gas_df = gas_df.reset_index()  # Ensure index is the date column

gas_df['us_gasprice_perc_change'] = 100 * (gas_df['us_gasprice'] - gas_df['us_gasprice'].shift(12))/gas_df['us_gasprice'].shift(12)
gas_df['padd1a_perc_change'] = 100 * (gas_df['padd1a_gasprice'] - gas_df['padd1a_gasprice'].shift(12))/gas_df['padd1a_gasprice'].shift(12)
gas_df['padd1b_perc_change'] = 100 * (gas_df['padd1b_gasprice'] - gas_df['padd1b_gasprice'].shift(12))/gas_df['padd1b_gasprice'].shift(12)
gas_df['padd1c_perc_change'] = 100 * (gas_df['padd1c_gasprice'] - gas_df['padd1c_gasprice'].shift(12))/gas_df['padd1c_gasprice'].shift(12)
gas_df['padd2_perc_change'] = 100 * (gas_df['padd2_gasprice'] - gas_df['padd2_gasprice'].shift(12))/gas_df['padd2_gasprice'].shift(12)
gas_df['padd3_perc_change'] = 100 * (gas_df['padd3_gasprice'] - gas_df['padd3_gasprice'].shift(12))/gas_df['padd3_gasprice'].shift(12)
gas_df['padd4_perc_change'] = 100 * (gas_df['padd4_gasprice'] - gas_df['padd4_gasprice'].shift(12))/gas_df['padd4_gasprice'].shift(12)
gas_df['padd5_perc_change'] = 100 * (gas_df['padd5_gasprice'] - gas_df['padd5_gasprice'].shift(12))/gas_df['padd5_gasprice'].shift(12)



# PADD colors for the map
padd_colors = {
    'PADD 1a': "#00FF7F",
    'PADD 1b': "#F2C14C",
    'PADD 1c': "#008000",
    'PADD 2': "#D83367",
    'PADD 3': "#6B3E6D",
    'PADD 4': "#A1B3D2",
    'PADD 5': "#008080",
}

padd_to_states = {
    'PADD 1a': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT'],
    'PADD 1b': ['DE', 'MD', 'NJ', 'NY', 'PA'],
    'PADD 1c': ['FL', 'GA', 'NC', 'SC', 'VA', 'WV'],
    'PADD 2': ['OK', 'IL', 'IN', 'IA', 'KS', 'KY', 'MI', 'MN', 'MO', 'NE', 'ND', 'OH', 'SD', 'TN', 'WI'],
    'PADD 3': ['AL', 'AR', 'LA', 'MS', 'NM', 'TX'],
    'PADD 4': ['CO', 'ID', 'MT', 'UT', 'WY'],
    'PADD 5': ['AK', 'AZ', 'CA', 'HI', 'NV', 'OR', 'WA'],
}

# Function to create the national map figure (with date filtering)
def make_national_fig(start_date=None, end_date=None):
    # Filter gas_df based on selected date range
    if start_date and end_date:
        mask = (gas_df['date'] >= start_date) & (gas_df['date'] <= end_date)
        filtered_df = gas_df[mask]
    else:
        filtered_df = gas_df

    # Get the last gas price within the filtered date range
    national_price = np.round(filtered_df['us_gasprice'].iloc[-1], 2) if not filtered_df.empty else 0
    yoy_change = np.round(filtered_df['us_gasprice_perc_change'].iloc[-1], 2) if not filtered_df.empty else 0

    
    df = pd.DataFrame({
        "usa": 'usa',
        "Price": [f' ${national_price}'],
        "1 Year Change": [f' {yoy_change}%']
    })
    
    fig = px.choropleth(
        df,
        locations="usa",
        locationmode="country names",
        color_discrete_sequence=["#FDBB84"],
        scope="usa",
        hover_data={
        "Price": True,
        "1 Year Change": True,
        "usa": False  # Hide this redundant field
    }
    )
    
    fig.update_layout(
        title="National Average Gas Price",
        geo=dict(showlakes=False),
        annotations=[dict(
            text=f"USA Avg: ${national_price:.2f}",
            x=0.5, y=0.5, xref='paper', yref='paper',
            showarrow=False, font=dict(size=24), bgcolor='white'
        )],
        clickmode="event+select"
    )
    
    return fig

# Function to create the PADD map figure with date filtering
def make_padd_fig(start_date=None, end_date=None):
    rows = []
    
    # Filter gas_df based on selected date range
    if start_date and end_date:
        mask = (gas_df['date'] >= start_date) & (gas_df['date'] <= end_date)
        filtered_df = gas_df[mask]
    else:
        filtered_df = gas_df
    
    # Get the last price for each PADD within the selected date range
    for padd, states in padd_to_states.items():
        last_price = filtered_df[f'{padd.lower().replace(" ", "")}_gasprice'].iloc[-1] if not filtered_df.empty else 0
        col_base = padd.lower().replace(" ", "")
        yoy_col = f"{col_base}_perc_change"
        yoy_change = np.round(filtered_df[yoy_col].iloc[-1], 2) if not filtered_df.empty else 0
        
        for state in states:
            rows.append({
                "state": state,
                "PADD": padd,
                "gas_price": last_price,
                "Price": f" ${last_price}",
                "1 Year Change": f" {yoy_change}%"
            })
    
    
    df = pd.DataFrame(rows)
    
    fig = px.choropleth(
        df,
        locations="state",
        locationmode="USA-states",
        color="PADD",
        hover_name="PADD",
        hover_data={"gas_price": False, "Price":True,  "state": False,  "1 Year Change": True, "PADD": True},
        scope="usa",
        color_discrete_map=padd_colors,
        title="Gas Price by PADD"
    )

    padd_coords = {
        "PADD 1a": (-70, 44), "PADD 1b": (-70, 36), "PADD 1c": (-74, 30),
        "PADD 2": (-90, 42), "PADD 3": (-95, 30), "PADD 4": (-110, 45), "PADD 5": (-120, 37),
    }

    fig.add_trace(go.Scattergeo(
        lon=[lon for lon, lat in padd_coords.values()],
        lat=[lat for lon, lat in padd_coords.values()],
        text=[f'{padd}: ${df.gas_price[df.PADD == padd].iloc[-1]:.2f}' for padd in padd_coords],
        mode="text+markers",
        textposition="top center",
        textfont=dict(size=12, color="black"),
        marker=dict(size=10, color="white", opacity=1),
        showlegend=False
    ))
    
   
    #print( [f'{padd}: ${df.gas_price[df.PADD == padd].loc[-1]}' for padd in padd_coords])

    fig.update_layout(
        geo=dict(scope="usa", showlakes=False, projection_type="albers usa"),
        clickmode="event+select"
    )
    
    return fig

# Function to create the line plot with date filtering
def make_lineplot(view_info, start_date=None, end_date=None):
    fig = go.Figure()

    # Filter data based on date range if provided
    if start_date and end_date:
        mask = (gas_df['date'] >= start_date) & (gas_df['date'] <= end_date)
        filtered_df = gas_df[mask]
    else:
        filtered_df = gas_df

    date_col = filtered_df['date']

    if view_info['level'] == 'national':
        fig.add_trace(go.Scatter(x=date_col, y=filtered_df['us_gasprice'], name='USA', line=dict(color='black')))
    elif view_info['level'] == 'padd':
        if view_info['selected']:
            padd_color = padd_colors.get(view_info['selected'], '#000000')  # Default to black if not found
            col = view_info['selected'].lower().replace(" ", "") + "_gasprice"
            if col in filtered_df.columns:
                fig.add_trace(go.Scatter(x=date_col, y=filtered_df[col], name=view_info['selected'],
                                         line=dict(color=padd_color)))  # Use the correct PADD color
        else:
            for padd in padd_colors.keys():
                col = padd.lower().replace(" ", "") + "_gasprice"
                padd_color = padd_colors.get(padd, '#000000')
                fig.add_trace(go.Scatter(x=date_col, y=filtered_df[col], name=padd,
                                         line=dict(color=padd_color)))

    fig.update_layout(
        height=300,
        margin=dict(l=50, r=10, t=40, b=40),
        title="Historical Gas Prices",
        xaxis_title="Date",
        yaxis_title="USD per gallon",
        hovermode="x unified"
    )
    return fig


app.layout = html.Div([
    html.H1("Gas Price Dashboard"),
    dcc.Store(id="map-level", data={"level": "national", "selected": None}),
    dcc.Graph(id="gas-map", figure=make_national_fig()),
    html.Button("Reset", id="back-button", n_clicks=0, style={"display": "none"}),
    dcc.Graph(id="line-plot"),
    
    # Add the Date Picker Range
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=gas_df['date'].min().strftime('%Y-%m-%d'),
        end_date=gas_df['date'].max().strftime('%Y-%m-%d'),
        display_format='YYYY-MM-DD',  # Display format
        style={'margin': '20px'}
    )
])


@app.callback(
    Output("back-button", "style"),
    Output("map-level", "data"),
    Output("date-picker-range", "start_date"),
    Output("date-picker-range", "end_date"),
    Input("gas-map", "clickData"),
    Input("back-button", "n_clicks"),
    State("map-level", "data"),
    prevent_initial_call=True
)
def drilldown(click_data, back_clicks, current_state):
    ctx = dash.callback_context

    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    # Full date range
    full_start = gas_df['date'].min().strftime('%Y-%m-%d')
    full_end = gas_df['date'].max().strftime('%Y-%m-%d')

    if trigger == "back-button":
        return (
            {"display": "none"},
            {"level": "national", "selected": None},
            full_start,
            full_end
        )

    elif trigger == "gas-map" and click_data:
        if current_state["level"] == "national":
            return (
                {"display": "inline-block"},
                {"level": "padd", "selected": None},
                dash.no_update,
                dash.no_update
            )
        elif current_state["level"] == "padd":
            clicked_padd = click_data["points"][0]["hovertext"]
            return (
                {"display": "inline-block"},
                {"level": "padd", "selected": clicked_padd},
                dash.no_update,
                dash.no_update
            )

    raise dash.exceptions.PreventUpdate


@app.callback(
    Output("line-plot", "figure"),
    [Input("map-level", "data"),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_line_plot(view_info, start_date, end_date):
    return make_lineplot(view_info, start_date, end_date)

@app.callback(
    Output("gas-map", "figure"),
    [Input("map-level", "data"),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_map(view_info, start_date, end_date):
    if view_info['level'] == 'national':
        return make_national_fig(start_date, end_date)
    else:
        return make_padd_fig(start_date, end_date)


if __name__ == "__main__":
    app.run(debug=True)

#%% Crude Oil Dashboard

'''
1. Line plot of total production
2. Plot of YoY change?
3. Line plot of total import
4. Line plot of total export
5. line plot of number of oil employees
6. bar plot of production by PADD to get clear idea of who produces most
'''
oil_series_cols = ['total_crude_oil_prod', 'padd1_crude_oil_prod', 'padd2_crude_oil_prod',
               'padd3_crude_oil_prod', 'padd4_crude_oil_prod', 'padd5_crude_oil_prod',
               'oil_employees', 'oil_price', 'us_gasprice', 'inflation_rate', 'cpi_all',
               'total_crude_oil_import', 'total_crude_oil_export']

oil_df = all_monthly[oil_series_cols]
oil_df.dropna(inplace = True)
oil_df.reset_index(inplace = True)  

oil_df['oil_price_perc_change'] = 100 * (oil_df['oil_price'] - oil_df['oil_price'].shift(12))/oil_df['oil_price'].shift(12)
oil_df['oil_prod_perc_change'] = 100 * (oil_df['total_crude_oil_prod'] - oil_df['total_crude_oil_prod'].shift(12))/oil_df['total_crude_oil_prod'].shift(12)
oil_df['oil_employee_perc_change'] = 100 * (oil_df['oil_employees'] - oil_df['oil_employees'].shift(12))/oil_df['oil_employees'].shift(12)
oil_df['oil_import_perc_change'] = 100 * (oil_df['total_crude_oil_import'] - oil_df['total_crude_oil_import'].shift(12))/oil_df['total_crude_oil_import'].shift(12)
oil_df['oil_export_perc_change'] = 100 * (oil_df['total_crude_oil_export'] - oil_df['total_crude_oil_export'].shift(12))/oil_df['total_crude_oil_export'].shift(12)


app = dash.Dash(__name__)

app.layout = html.Div([
    
    html.H2("Oil Dashboard"),
    
    dcc.DatePickerRange(
        id='date-picker',
        min_date_allowed=oil_df['date'].min(),
        max_date_allowed=oil_df['date'].max(),
        start_date=oil_df['date'].min(),
        end_date=oil_df['date'].max()
    ),

    html.Div([
        html.Div([dcc.Graph(id='prod_fig')], style={'width': '100%', 'display': 'inline-block'})
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),

    html.Div([
        html.Div([dcc.Graph(id='bar-plot')], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([dcc.Graph(id='price-plot')], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ], style={'width': '100%', 'paddingTop': '20px'}),
    
    html.Div([
        html.Div([dcc.Graph(id='import_fig')], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([dcc.Graph(id='export_fig')], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ], style={'width': '100%', 'paddingTop': '20px'}),

    html.Div([
        html.Div([dcc.Graph(id='emp_fig')], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([dcc.Graph(id='heatmap')], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ], style={'width': '100%', 'paddingTop': '20px'})
    
])

@app.callback(
    [Output('prod_fig', 'figure'),
     Output('emp_fig', 'figure'),
     Output('bar-plot', 'figure'),
     Output('price-plot', 'figure'),
     Output('heatmap', 'figure'),
     Output('import_fig', 'figure'),
     Output('export_fig', 'figure')],
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_graphs(start_date, end_date):
    # Filter data
    mask = (oil_df['date'] >= start_date) & (oil_df['date'] <= end_date)
    filtered_df = oil_df.loc[mask]

    ### crude oil production plot ###
    prod_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],  # 70% line plot, 30% bar plot
        vertical_spacing=0.02,
        subplot_titles=("Crude Oil Production", " ")
    )
    #production
    prod_fig.add_trace(
        go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['total_crude_oil_prod'],
            name="Crude Oil Production",
            mode='lines',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    #YoY Change
    bar_colors = filtered_df['oil_prod_perc_change']
    prod_fig.add_trace(
        go.Bar(
            x=filtered_df['date'],
            y=filtered_df['oil_prod_perc_change'],
            marker=dict(
                color=bar_colors,
                colorscale='RdYlGn',  # red-yellow-green
                cmin=bar_colors.min(),
                cmax=bar_colors.max(),
                colorbar=dict(title="YoY Change"),
                showscale = False
            ),
        ),
        row=2, col=1
    )
    prod_fig.update_layout(
        height=500,
        showlegend=False,
        title="Crude Oil Production with YoY Change",
        xaxis2=dict(title="Date"),  # applies to shared x-axis
        yaxis=dict(title="Production (MMBL/Day)"),
        yaxis2=dict(title="YoY Change (%)"),
    )


    ### oil employees plot ###
    emp_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],  # 70% line plot, 30% bar plot
        vertical_spacing=0.02,
        subplot_titles=("Oil Industry Employees", " ")
    )
    #counts
    emp_fig.add_trace(
        go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['oil_employees'],
            name="Oil Employees",
            mode='lines',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    #YoY Change
    bar_colors = filtered_df['oil_employee_perc_change']
    emp_fig.add_trace(
        go.Bar(
            x=filtered_df['date'],
            y=filtered_df['oil_employee_perc_change'],
            marker=dict(
                color=bar_colors,
                colorscale='RdYlGn',  # red-yellow-green
                cmin=bar_colors.min(),
                cmax=bar_colors.max(),
                colorbar=dict(title="YoY Change"),
                showscale = False
            ),
        ),
        row=2, col=1
    )
    emp_fig.update_layout(
        height=500,
        showlegend=False,
        title="Oil Industry Employees with YoY Change",
        xaxis2=dict(title="Date"),  # applies to shared x-axis
        yaxis=dict(title="Count (Thousands)"),
        yaxis2=dict(title="YoY Change (%)"),
    )
    
    ### Bar plot - Latest production by PADD ###
    if not filtered_df.empty:
        latest = filtered_df.iloc[-1]
        bar_data = pd.DataFrame({
            'area': ['PADD 1', 'PADD 2', 'PADD 3',
                     'PADD 4', 'PADD 5'],
            'production': [latest['padd1_crude_oil_prod'], latest['padd2_crude_oil_prod'], 
                           latest['padd3_crude_oil_prod'], latest['padd4_crude_oil_prod'],
                           latest['padd5_crude_oil_prod']]
        })
    else:
        bar_data = pd.DataFrame({'area': [], 'production': []})

    bar_fig = px.bar(bar_data, x='area', y='production', title="Production by PADD")
    bar_fig.update_layout(
        xaxis_title="PADD",
        yaxis_title="Production (MMBL/Day)",  
       
    )
    
    ### Price Plot ###
    price_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],  # 70% line plot, 30% bar plot
        vertical_spacing=0.02,
        subplot_titles=("Oil Price (West Texas Intermediate)", " ")
    )
    #production
    price_fig.add_trace(
        go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['oil_price'],
            name="Oil Price",
            mode='lines',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    #YoY Change
    bar_colors = filtered_df['oil_price_perc_change']
    price_fig.add_trace(
        go.Bar(
            x=filtered_df['date'],
            y=filtered_df['oil_price_perc_change'],
            marker=dict(
                color=bar_colors,
                colorscale='RdYlGn',  # red-yellow-green
                cmin=bar_colors.min(),
                cmax=bar_colors.max(),
                colorbar=dict(title="YoY Change"),
                showscale = False
            ),
        ),
        row=2, col=1
    )
    price_fig.update_layout(
        height=500,
        showlegend=False,
        title="Crude Oil Price with YoY Change",
        xaxis2=dict(title="Date"),  # applies to shared x-axis
        yaxis=dict(title="Price Per Barrel"),
        yaxis2=dict(title="YoY Change (%)"),
    )
    
    ### Correlation Heatmap ###
    
    filtered_df2 = filtered_df.copy()
    filtered_df2 = filtered_df2[['total_crude_oil_prod', 'oil_prod_perc_change',
                   'oil_employees', 'oil_employee_perc_change', 'total_crude_oil_import', 
                    'total_crude_oil_export', 'oil_price', 'us_gasprice', 'cpi_all',
                   'inflation_rate']]
    
    filtered_df2.columns = ['oil production', 'prodction YoY',
                   'employees', 'employees YoY', 'imports', 'exports', 
                   'price', 'gas price', 'cpi', 'inflation rate']
    
    corr_matrix = filtered_df2.corr(numeric_only=True)
    heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation")
    ))

    heatmap.update_layout(
        title="Pearson Correlations",
        #xaxis_title="Variables",
        #yaxis_title="Variables",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        width=500,
        height=500
    )
    
    ### Import Plot ###

    import_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],  # 70% line plot, 30% bar plot
        vertical_spacing=0.02,
        subplot_titles=("Oil Imports", " ")
    )
    #production
    import_fig.add_trace(
        go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['total_crude_oil_import'],
            name="Oil Imports",
            mode='lines',
            line=dict(color='purple')
        ),
        row=1, col=1
    )
    #YoY Change
    bar_colors = filtered_df['oil_import_perc_change']
    import_fig.add_trace(
        go.Bar(
            x=filtered_df['date'],
            y=filtered_df['oil_import_perc_change'],
            marker=dict(
                color=bar_colors,
                colorscale='RdYlGn',  # red-yellow-green
                cmin=bar_colors.min(),
                cmax=bar_colors.max(),
                colorbar=dict(title="YoY Change"),
                showscale = False
            ),
        ),
        row=2, col=1
    )
    import_fig.update_layout(
        height=500,
        showlegend=False,
        title="Crude Oil Imports with YoY Change",
        xaxis2=dict(title="Date"),  # applies to shared x-axis
        yaxis=dict(title="Mega Barrels of Import"),
        yaxis2=dict(title="YoY Change (%)"),
    )
    
    ### Export Plot ###

    export_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],  # 70% line plot, 30% bar plot
        vertical_spacing=0.02,
        subplot_titles=("Oil Exports", " ")
    )
    #production
    export_fig.add_trace(
        go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['total_crude_oil_export'],
            name="Oil Exports",
            mode='lines',
            line=dict(color='darkgreen')
        ),
        row=1, col=1
    )
    #YoY Change
    bar_colors = filtered_df['oil_export_perc_change']
    export_fig.add_trace(
        go.Bar(
            x=filtered_df['date'],
            y=filtered_df['oil_export_perc_change'],
            marker=dict(
                color=bar_colors,
                colorscale='RdYlGn',  # red-yellow-green
                cmin=bar_colors.min(),
                cmax=bar_colors.max(),
                colorbar=dict(title="YoY Change"),
                showscale = False
            ),
        ),
        row=2, col=1
    )
    export_fig.update_layout(
        height=500,
        showlegend=False,
        title="Crude Oil Exports with YoY Change",
        xaxis2=dict(title="Date"),  # applies to shared x-axis
        yaxis=dict(title="Mega Barrels of Export"),
        yaxis2=dict(title="YoY Change (%)"),
    )

    return prod_fig, emp_fig, bar_fig, price_fig, heatmap, import_fig, export_fig

if __name__ == "__main__":
    app.run(debug=True)

#%% Industrial Dashboard

industry_cols = [
    'cpi_all', 'inflation_rate',
    'manufact_employee',
    #'all_construction_spend',
    #'manufact_construction_spend',
    'housing_under_construction',
    'neast_housing_under_construction',
    'west_housing_under_construction',
    'south_housing_under_construction',
    'mwest_housing_under_construction'
]

industry_cols_price_adj = [
    'total_manufact_orders',
    'durable_manufact_orders',
    'machinery_manufact_orders',
    'consumer_goods_manufact_orders',
    'vehicle_manufact_orders',
    'construction_materials_manufact_orders',
    'computer_manufact_orders',
    'furnature_manufact_orders',
    'capital_manufact_orders',
    'transport_manufact_orders',
    'it_manufact_orders',
    'transport_output',
    'all_output',
    'manufact_output',
    'construction_output',
    'buisness_service_output',
    'retail_output',
    'agriculture_output',
    'finance_output',
    'mining_output',
    'it_output',
    'utilities_output',
    'gov_output',
    'other_output'
    ]

cpi_2016 = all_monthly['cpi_all'][all_monthly.index == '2016-01-31'].values[0]
ind_df = all_monthly[industry_cols + industry_cols_price_adj]
ind_df.reset_index(inplace = True)
ind_df = ind_df[ind_df['date'] >= '1995-01-01']

#adjust these to 2016 CPI 
for col in industry_cols_price_adj:
    ind_df[col] = ind_df[col] * (ind_df['cpi_all']/cpi_2016)

#add total perc changes YoY
ind_df['total_manufact_orders_perc_change'] = 100 * (ind_df['total_manufact_orders'] - ind_df['total_manufact_orders'].shift(12))/ind_df['total_manufact_orders'].shift(12)
ind_df['all_output_perc_change'] = 100 * (ind_df['all_output'] - ind_df['all_output'].shift(12))/ind_df['all_output'].shift(12)
ind_df['manufact_employee_perc_change'] = 100 * (ind_df['manufact_employee'] - ind_df['manufact_employee'].shift(12))/ind_df['manufact_employee'].shift(12)
ind_df['housing_perc_change'] = 100 * (ind_df['housing_under_construction'] - ind_df['housing_under_construction'].shift(12))/ind_df['housing_under_construction'].shift(12)

clean_orders_names = {
    'durable_manufact_orders' : 'Durable Goods',
     'machinery_manufact_orders': 'Machinery',
     'consumer_goods_manufact_orders': 'Consumer Goods',
     'vehicle_manufact_orders': 'Vehicles',
     'construction_materials_manufact_orders': 'Construction Materials',
     'computer_manufact_orders': 'Computers',
     'furnature_manufact_orders': 'Furnature',
     'capital_manufact_orders': 'Captial Goods',
     'transport_manufact_orders' : 'Transportation Goods',
     'it_manufact_orders': 'IT'
}

clean_output_names = {
    'manufact_output': 'Manufacturing',
     'construction_output': 'Construction',
     'buisness_service_output': 'Buisness Service',
     'retail_output': 'Retail',
     'agriculture_output': 'Agriculture',
     'finance_output': 'Finance',
     'mining_output': 'Mining',
     'it_output': 'IT',
     'utilities_output': 'Utilities',
     'gov_output': 'Government',
     'other_output': 'Other'
}


clean_housing_names = {
    'housing_under_construction': 'Total',
    'neast_housing_under_construction': 'North East',
    'west_housing_under_construction': 'West',
    'south_housing_under_construction': 'South',
    'mwest_housing_under_construction': 'Midwest'
    }

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Industry Dashboard"),

    dcc.DatePickerRange(
        id='date-picker',
        min_date_allowed=ind_df['date'].min(),
        max_date_allowed=ind_df['date'].max(),
    ),
    
    html.Button("Reset", id="reset-button", n_clicks=0, style={'margin': '20px 0'}),
    
    html.Div([
        html.Div([dcc.Graph(id='industry_output')], style={'width': '100%'}),
    ]),

    html.Div([
        html.Label("Select Output Subcategories:"),
        dcc.Dropdown(
            id='subcategory-output-dropdown',
            options=[{'label': clean_output_names[cat], 'value': cat} for cat in list(clean_output_names.keys())],
            multi=True,
            value=list(clean_output_names.keys()) # starts with no selection
        )
    ], style={'margin': '20px 0'}),

    html.Div([
        html.Div([dcc.Graph(id='sub_output')], style={'width': '100%'}, id='sub_output_container')
    ]),
    
    html.Div(
        [dcc.Graph(id='manufact_orders')], style={'width': '100%'}
        ),
    

    html.Div([
        html.Label("Select Orders Subcategories:"),
        dcc.Dropdown(
            id='subcategory-orders-dropdown',
            options=[{'label': clean_orders_names[cat], 'value': cat} for cat in list(clean_orders_names.keys())],
            multi=True,
            value=list(clean_orders_names.keys())   # starts with no selection
        )
    ], style={'margin': '20px 0'}),

    html.Div([
        html.Div([dcc.Graph(id='sub_orders')], style={'width': '100%'}, id='sub_orders_container')
    ]),
    
    html.Div(
        [dcc.Graph(id='housing')], style={'width': '100%'}
        ),
    
    html.Div([
        html.Label("Select Housing Subregion:"),
        dcc.Dropdown(
            id='subregion-housing-dropdown',
            options=[{'label': clean_housing_names[cat], 'value': cat} for cat in list(clean_housing_names.keys())],
            multi=True,
            value= list(set(clean_housing_names.keys()) - set(['housing_under_construction']))   # starts with no selection
        )
    ], style={'margin': '20px 0'}),

    html.Div([
        html.Div([dcc.Graph(id='sub_housing')], style={'width': '100%'}, id='sub_housing_container')
    ]),
    
    html.Div([
        html.Div([dcc.Graph(id='man_emp_fig')], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        html.Div([dcc.Graph(id='heatmap')], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    ], style={'width': '100%', 'paddingTop': '20px'})
    
])


@app.callback(
    [
     Output('industry_output', 'figure'),
     Output('sub_output', 'figure'),
     Output('manufact_orders', 'figure'),
     Output('sub_orders', 'figure'),
     Output('housing', 'figure'),
     Output('sub_housing', 'figure'),
     Output('heatmap', 'figure'),
     Output('man_emp_fig', 'figure'),
     Output('date-picker', 'start_date'),
     Output('date-picker', 'end_date'),
     ],
    
    [Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('subcategory-output-dropdown', 'value'),
     Input('subcategory-orders-dropdown', 'value'),
     Input('subregion-housing-dropdown', 'value'),
     Input('reset-button', 'n_clicks')
     
     ]
)
def update_graphs(start_date, end_date, selected_output_subcategories, selected_orders_subcategories, selected_housing_subregions, reset_clicks):
    #set defaults for reset
    triggered_id = ctx.triggered_id
    
    # Set default values
    default_start = ind_df['date'].min()
    default_end = ind_df['date'].max()
    default_output = industry_cols_price_adj[-11:]
    default_orders = industry_cols_price_adj[1:11]
    default_regions = list(clean_housing_names.keys())
    if start_date is None:
        start_date = default_start
    if end_date is None:
        end_date = default_end

    if triggered_id == 'reset-button':
        start_date = default_start
        end_date = default_end
        selected_output_subcategories = default_output
        selected_orders_subcategories = default_orders
        selected_housing_subregions = default_regions
    
    # Filter data
    mask = (ind_df['date'] >= start_date) & (ind_df['date'] <= end_date)
    filtered_df = ind_df.loc[mask]
    
    
    ### Total Industry Output Plot ###
    
    output_df = filtered_df.copy()
    output_df = output_df[['date', 'all_output_perc_change', 'all_output'] + industry_cols_price_adj[-11:]]
    output_df.dropna(inplace=True)

    
    industry_output = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.02,
        subplot_titles=("Total Industry Output", " ")
    )
    industry_output.add_trace(
        go.Scatter(
            x=output_df['date'],
            y=output_df['all_output'],
            name="Total Industry Output",
            mode='lines',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    bar_colors = output_df['all_output_perc_change']
    industry_output.add_trace(
        go.Bar(
            x=output_df['date'],
            y=output_df['all_output_perc_change'],
            marker=dict(
                color=bar_colors,
                colorscale='RdYlGn',
                cmin=bar_colors.min(),
                cmax=bar_colors.max(),
                showscale=False
            ),
        ),
        row=2, col=1
    )
    industry_output.update_layout(
        height=500,
        showlegend=False,
        title="Total Industry Output with YoY Change",
        xaxis2=dict(title="Date"),
        yaxis=dict(title="Output (Billions $)"),
        yaxis2=dict(title="YoY Change (%)"),
    )
    
    ### Subcategory Output Plot ###
    sub_output = go.Figure()

    # Check if user clicked on the main plot (drilldown logic)
        
    if selected_output_subcategories:
        for cat in selected_output_subcategories:
            sub_output.add_trace(go.Scatter(
                x=output_df['date'],
                y=output_df[cat],
                mode='lines',
                name=clean_output_names.get(cat, cat)
            ))
            
    sub_output.update_layout(
        title="Industry Output by Category",
        xaxis_title="Date",
        yaxis_title="Output (Billions $)",
        height=400
    )
    
    
    ### Total Manufacturing Orders Plot ###
    manufact_orders = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.02,
        subplot_titles=("Total New Manufacturing Orders", " ")
    )
    manufact_orders.add_trace(
        go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['total_manufact_orders'],
            name="Total New Manufacturing Orders",
            mode='lines',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    bar_colors = filtered_df['total_manufact_orders_perc_change']
    manufact_orders.add_trace(
        go.Bar(
            x=filtered_df['date'],
            y=filtered_df['total_manufact_orders_perc_change'],
            marker=dict(
                color=bar_colors,
                colorscale='RdYlGn',
                cmin=bar_colors.min(),
                cmax=bar_colors.max(),
                showscale=False
            ),
        ),
        row=2, col=1
    )
    manufact_orders.update_layout(
        height=500,
        showlegend=False,
        title="Total New Manufacturing Orders with YoY Change",
        xaxis2=dict(title="Date"),
        yaxis=dict(title="New Orders (Millions $)"),
        yaxis2=dict(title="YoY Change (%)"),
    )

    ### Subcategory Output Plot ###
    sub_orders = go.Figure()

    # Check if user clicked on the main plot (drilldown logic)w
    if selected_orders_subcategories:
        for cat in selected_orders_subcategories:
            sub_orders.add_trace(go.Scatter(
                x=filtered_df['date'],
                y=filtered_df[cat],
                mode='lines',
                name=clean_orders_names.get(cat, cat)
            ))


    sub_orders.update_layout(
        title="Manufacturing Orders by Category",
        xaxis_title="Date",
        yaxis_title="Orders (Millions $)",
        height=400
    )
    
    
    ### Total Construction Plot ###
    housing = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.02,
        subplot_titles=("New Housing Units Under Construction", " ")
    )
    housing.add_trace(
        go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['housing_under_construction'],
            name="New Housing Units Under Construction",
            mode='lines',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    bar_colors = filtered_df['housing_perc_change']
    housing.add_trace(
        go.Bar(
            x=filtered_df['date'],
            y=filtered_df['housing_perc_change'],
            marker=dict(
                color=bar_colors,
                colorscale='RdYlGn',
                cmin=bar_colors.min(),
                cmax=bar_colors.max(),
                showscale=False
            ),
        ),
        row=2, col=1
    )
    housing.update_layout(
        height=500,
        showlegend=False,
        title="New Housing Units Under Construction with YoY Change",
        xaxis2=dict(title="Date"),
        yaxis=dict(title="New Unites (Thousands)"),
        yaxis2=dict(title="YoY Change (%)"),
    )
    ### Subregion Housing Plot ###
    sub_housing = go.Figure()

    # Check if user clicked on the main plot (drilldown logic)w
    if selected_housing_subregions:
        for cat in selected_housing_subregions:
            sub_housing.add_trace(go.Scatter(
                x=filtered_df['date'],
                y=filtered_df[cat],
                mode='lines',
                name=clean_housing_names.get(cat, cat)
            ))


    sub_housing.update_layout(
        title="New Housing by Census Region",
        xaxis_title="Date",
        yaxis_title="New Units (Thousands)",
        height=400
    )
    
    ### Correlation Heatmap ###
    
    heatmap_df = filtered_df.copy()
    heatmap_df = heatmap_df[['cpi_all', 'inflation_rate', 'manufact_employee', 'manufact_employee_perc_change',
                               'housing_under_construction', 'housing_perc_change', 'total_manufact_orders', 
                               'total_manufact_orders_perc_change',
                               'all_output', 'all_output_perc_change']]
    
    heatmap_df.columns = ['cpi', 'inflation rate', 'manufacturing employees', 'employee % change',
                               'housing construction', 'housing % change', 'manufacturing orders',
                               'manufacturing % change', 
                               'industry output', 'output % change']
    
    corr_matrix = heatmap_df.corr(numeric_only=True)
    heatmap = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation")
    ))

    heatmap.update_layout(
        title="Pearson Correlations",
        #xaxis_title="Variables",
        #yaxis_title="Variables",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        width=500,
        height=500
    )
    ### Manufacturing Employees ###
    man_emp_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],  # 70% line plot, 30% bar plot
        vertical_spacing=0.02,
        subplot_titles=("Manufacturing Employees", " ")
    )
    #counts
    man_emp_fig.add_trace(
        go.Scatter(
            x=filtered_df['date'],
            y=filtered_df['manufact_employee'],
            name="Oil Employees",
            mode='lines',
            line=dict(color='royalblue')
        ),
        row=1, col=1
    )
    #YoY Change
    bar_colors = filtered_df['manufact_employee_perc_change']
    man_emp_fig.add_trace(
        go.Bar(
            x=filtered_df['date'],
            y=filtered_df['manufact_employee_perc_change'],
            marker=dict(
                color=bar_colors,
                colorscale='RdYlGn',  # red-yellow-green
                cmin=bar_colors.min(),
                cmax=bar_colors.max(),
                colorbar=dict(title="YoY Change"),
                showscale = False
            ),
        ),
        row=2, col=1
    )
    man_emp_fig.update_layout(
        height=500,
        showlegend=False,
        title="Manufacturing Employees with YoY Change",
        xaxis2=dict(title="Date"),  # applies to shared x-axis
        yaxis=dict(title="Count (Thousands)"),
        yaxis2=dict(title="YoY Change (%)"),
    )
    
    
    return industry_output, sub_output, manufact_orders, sub_orders, housing, sub_housing, heatmap, man_emp_fig, start_date, end_date



if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    
#%% Employment Dashboard

'''
1. line plot unemployement rate
2. line plot of % non-farm payroll employees by different sectors (new data needed for all)
    a. manufacturing
    b. oil
    c. tech
    d. finance
    e. service sector
    f. retail
    g. others?
3. wage plots overtime (all need new data)
    a) regions too?
    b) sectors?
4. number of govt workers   
5. plots of % eligible workers by age group (new data on FRED required)
 

'''


#%% Overall/Summary Dashboard

'''
36 month plots, see if clean way to overlay YoY changes?

1. industrial production ?
2. imports
3. exports
4. unemployment rate
5. inflation
6. government spending
7. Stock index or 2
8. Correlation plots
9. Clustering?
'''
