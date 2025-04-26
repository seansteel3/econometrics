#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

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

clean_padd_names = {
    'padd1a_gasprice': 'PADD 1a',
    'padd1b_gasprice': 'PADD 1b',
    'padd1c_gasprice': 'PADD 1c',
    'padd2_gasprice': 'PADD 2',
    'padd3_gasprice': 'PADD 3',
    'padd4_gasprice': 'PADD 4',
    'padd5_gasprice': 'PADD 5',
     
}

def get_data(all_monthly):
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
    return gas_df 

def get_layout(all_monthly):
    
    gas_df = get_data(all_monthly)


    layout = html.Div([
        html.H2("Gas Price Dashboard"),
        
        html.Button("Reset", id="reset-button-gas", n_clicks=0, style={'margin': '20px 0'}),
    
        dcc.DatePickerRange(
            id='date-picker-gas',
            min_date_allowed=gas_df['date'].min(),
            max_date_allowed=gas_df['date'].max(),
            start_date=gas_df['date'].min(),
            end_date=gas_df['date'].max()
        ),
        
        html.Div([
            html.Div([dcc.Graph(id='padd-fig')], style={'width': '100%', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
    
        html.Div([
            html.Div([dcc.Graph(id='us-gas-price-plot')], style={'width': '100%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='gas-price-plot')], style={'width': '100%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
    ])
    return layout


def register_callbacks(app, all_monthly):
    
    gas_df = get_data(all_monthly)
    

    @app.callback(
        [Output('padd-fig', 'figure'),
         Output('gas-price-plot', 'figure'),
         Output('us-gas-price-plot', 'figure'),
         #Output('heatmap-gas', 'figure'),
         Output('date-picker-gas', 'start_date'),
         Output('date-picker-gas', 'end_date')
         ],
        [Input('date-picker-gas', 'start_date'),
         Input('date-picker-gas', 'end_date'),
         Input('reset-button-gas', 'n_clicks')]
    )
    
    
    def update_graphs(start_date, end_date, reset_clicks):
        #set defaults for reset
        triggered_id = ctx.triggered_id
        
        # Set default values
        default_start = gas_df['date'].min()
        default_end = gas_df['date'].max()
        if start_date is None:
            start_date = default_start
        if end_date is None:
            end_date = default_end
    
        if triggered_id == 'reset-button-gas':
            start_date = default_start
            end_date = default_end
        
        # Filter data
        mask = (gas_df['date'] >= start_date) & (gas_df['date'] <= end_date)
        filtered_df = gas_df.loc[mask]
        
        rows = []
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
        
        padd_fig = px.choropleth(
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
    
        padd_fig.add_trace(go.Scattergeo(
            lon=[lon for lon, lat in padd_coords.values()],
            lat=[lat for lon, lat in padd_coords.values()],
            text=[f'{padd}: ${df.gas_price[df.PADD == padd].iloc[-1]:.2f}' for padd in padd_coords],
            mode="text+markers",
            textposition="top center",
            textfont=dict(size=12, color="black"),
            marker=dict(size=10, color="white", opacity=1),
            showlegend=False
        ))
        
        padd_fig.update_layout(
            geo=dict(scope="usa", showlakes=False, projection_type="albers usa"),
            clickmode="event+select"
        )
        
        ### Price line plot ###
        gas_price_fig = go.Figure()
    
        # Filter data based on date range if provided
        date_col = filtered_df['date']
            
        for padd in padd_colors.keys():
            col = padd.lower().replace(" ", "") + "_gasprice"
            padd_color = padd_colors.get(padd, '#000000')
            gas_price_fig.add_trace(go.Scatter(x=date_col, y=filtered_df[col], name=padd,
                                     line=dict(color=padd_color)))
    
        gas_price_fig.update_layout(
            height=300,
            margin=dict(l=50, r=10, t=40, b=40),
            title="Historical Gas Prices",
            xaxis_title="Date",
            yaxis_title="USD per gallon",
            hovermode="x unified"
        )
        
        ### US price line plot ###
        us_gas_price_fig = go.Figure()
    
        # Filter data based on date range if provided
        date_col = filtered_df['date']
        print(filtered_df['us_gasprice'])
        us_gas_price_fig.add_trace(go.Scatter(x=date_col, y=filtered_df['us_gasprice'], name='US avg Gas Price',
                                 line=dict(color='royalblue')))

        us_gas_price_fig.update_layout(
            height=300,
            margin=dict(l=50, r=10, t=40, b=40),
            title="Average US Gas Price",
            xaxis_title="Date",
            yaxis_title="USD per gallon",
            hovermode="x unified"
        )
        
        
        return padd_fig, gas_price_fig, us_gas_price_fig, start_date, end_date
    
    


