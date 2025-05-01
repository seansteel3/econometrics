#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from fredapi import Fred
#import eia as Eia 
import requests
from functools import reduce
import time
import datetime
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from tqdm import tqdm

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboards.industrial_dash import get_layout as get_layout1
from dashboards.industrial_dash import register_callbacks as register_callbacks1
from dashboards.oil_dash import get_layout as get_layout2
from dashboards.oil_dash import register_callbacks as register_callbacks2
from dashboards.price_dash import get_layout as get_layout3
from dashboards.price_dash import register_callbacks as register_callbacks3
from dashboards.gas_dash import get_layout as get_layout4
from dashboards.gas_dash import register_callbacks as register_callbacks4
from dashboards.employ_dash import get_layout as get_layout5
from dashboards.employ_dash import register_callbacks as register_callbacks5
from dashboards.correlation_dash import get_layout as get_layout6
from dashboards.correlation_dash import register_callbacks as register_callbacks6

def main():
    
    print("Loading data (APIs can take up to 1min)...")

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
    for i in tqdm(range(len(source_df))):
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

    #%% Load the dashboards themselves

    print("Building Report...")

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    
    
    app.layout = html.Div([
        dcc.Tabs(id="tabs", value='tab1', children=[
            dcc.Tab(label='Industrial Report', value='tab1'),
            dcc.Tab(label='Oil Report', value='tab2'),
            dcc.Tab(label='Price Report', value='tab3'),
            dcc.Tab(label='Gas Price Report', value='tab4'),
            dcc.Tab(label='Employment Report', value='tab5'),
            dcc.Tab(label='Correlation Report', value='tab6'),
             #Add more tabs
        ]),
        html.Div(id='tab-content')
    ])
    
    @app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
    def render_content(tab):
        if tab == 'tab1':
            return get_layout1(all_monthly)
        elif tab == 'tab2':
            return get_layout2(all_monthly)
        elif tab == 'tab3':
            return get_layout3(all_monthly)
        elif tab == 'tab4':
            return get_layout4(all_monthly)
        elif tab == 'tab5':
            return get_layout5(all_monthly)
        elif tab == 'tab6':
            return get_layout6(all_monthly)
        
    register_callbacks1(app, all_monthly)
    register_callbacks2(app, all_monthly)
    register_callbacks3(app, all_monthly)
    register_callbacks4(app, all_monthly)
    register_callbacks5(app, all_monthly)
    register_callbacks6(app, all_monthly)


    
    app.run(debug=False)


if __name__ == '__main__':
    main()
