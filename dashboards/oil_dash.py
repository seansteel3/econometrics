#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from functs import make_lineplot_with_yoybars, generate_basic_heatmap



def get_data(all_monthly):
    oil_series_cols = ['total_crude_oil_prod', 'padd1_crude_oil_prod', 'padd2_crude_oil_prod',
                       'padd3_crude_oil_prod', 'padd4_crude_oil_prod', 'padd5_crude_oil_prod',
                       'oil_employees', 'oil_price', 'us_gasprice', 'inflation_rate', 'cpi_all',
                       'total_crude_oil_import', 'total_crude_oil_export']

    oil_df = all_monthly[oil_series_cols].copy()
    oil_df.dropna(inplace=True)
    oil_df.reset_index(inplace=True)

    # Derived columns
    oil_df['oil_price_perc_change'] = 100 * (oil_df['oil_price'] - oil_df['oil_price'].shift(12)) / oil_df['oil_price'].shift(12)
    oil_df['oil_prod_perc_change'] = 100 * (oil_df['total_crude_oil_prod'] - oil_df['total_crude_oil_prod'].shift(12)) / oil_df['total_crude_oil_prod'].shift(12)
    oil_df['oil_employee_perc_change'] = 100 * (oil_df['oil_employees'] - oil_df['oil_employees'].shift(12)) / oil_df['oil_employees'].shift(12)
    oil_df['oil_import_perc_change'] = 100 * (oil_df['total_crude_oil_import'] - oil_df['total_crude_oil_import'].shift(12)) / oil_df['total_crude_oil_import'].shift(12)
    oil_df['oil_export_perc_change'] = 100 * (oil_df['total_crude_oil_export'] - oil_df['total_crude_oil_export'].shift(12)) / oil_df['total_crude_oil_export'].shift(12)
    return oil_df

def get_layout(all_monthly):
    
    oil_df = get_data(all_monthly)
    
    layout = html.Div([
        html.H2("Oil Dashboard"),
        
        html.Button("Reset", id="reset-button-oil", n_clicks=0, style={'margin': '20px 0'}),

        dcc.DatePickerRange(
            id='date-picker-oil',
            min_date_allowed=oil_df['date'].min(),
            max_date_allowed=oil_df['date'].max(),
            start_date=oil_df['date'].min(),
            end_date=oil_df['date'].max()
        ),

        html.Div([
            html.Div([dcc.Graph(id='prod_fig')], style={'width': '100%', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),

        html.Div([
            html.Div([dcc.Graph(id='bar-plot')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='price-plot')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='import_fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='export_fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),

        html.Div([
            html.Div([dcc.Graph(id='emp_fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='heatmap-oil')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'})
    ])

    return layout





def register_callbacks(app, all_monthly):
    
    oil_df = get_data(all_monthly)
    
    @app.callback(
        [Output('prod_fig', 'figure'),
         Output('emp_fig', 'figure'),
         Output('bar-plot', 'figure'),
         Output('price-plot', 'figure'),
         Output('heatmap-oil', 'figure'),
         Output('import_fig', 'figure'),
         Output('export_fig', 'figure'),
         Output('date-picker-oil', 'start_date'),
         Output('date-picker-oil', 'end_date'),],
        [Input('date-picker-oil', 'start_date'),
         Input('date-picker-oil', 'end_date'),
         Input('reset-button-oil', 'n_clicks')]
    )


    def update_graphs(start_date, end_date, reset_clicks):
        #set defaults for reset
        triggered_id = ctx.triggered_id
        
        # Set default values
        default_start = oil_df['date'].min()
        default_end = oil_df['date'].max()
        if start_date is None:
            start_date = default_start
        if end_date is None:
            end_date = default_end
    
        if triggered_id == 'reset-button-oil':
            start_date = default_start
            end_date = default_end
        
        
        # Filter data
        mask = (oil_df['date'] >= start_date) & (oil_df['date'] <= end_date)
        filtered_df = oil_df.loc[mask]
    
        ### crude oil production plot ###
        prod_fig = make_lineplot_with_yoybars(filtered_df, main_data= 'total_crude_oil_prod',
                                               yoy_data = 'oil_prod_perc_change',
                                               title = 'Crude Oil Production',
                                               yaxis_label= 'Production (MMBL/Day)',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        

    
    
        ### oil employees plot ###
        emp_fig = make_lineplot_with_yoybars(filtered_df, main_data= 'oil_employees',
                                               yoy_data = 'oil_employee_perc_change',
                                               title = 'Oil Industry Employees',
                                               yaxis_label= 'Count (Thousands)',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        
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
        price_fig = make_lineplot_with_yoybars(filtered_df, main_data= 'oil_price',
                                               yoy_data = 'oil_price_perc_change',
                                               title = 'Oil Price (West Texas Intermediate)',
                                               yaxis_label= 'Price Per Barrel',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        
        ### Correlation Heatmap ###
        columns_heatmap = ['total_crude_oil_prod', 'oil_prod_perc_change',
                       'oil_employees', 'oil_employee_perc_change', 'total_crude_oil_import', 
                        'total_crude_oil_export', 'oil_price', 'us_gasprice', 'cpi_all',
                       'inflation_rate']
        
        names_heatmap = ['oil production', 'prodction YoY',
                       'employees', 'employees YoY', 'imports', 'exports', 
                       'price', 'gas price', 'cpi', 'inflation rate']
        
        heatmap = generate_basic_heatmap(filtered_df, columns_heatmap, names_heatmap)
        
        ### Import Plot ###
    
        import_fig = make_lineplot_with_yoybars(filtered_df, main_data= 'total_crude_oil_import',
                                               yoy_data = 'oil_import_perc_change',
                                               title = 'Oil Imports',
                                               yaxis_label= 'Mega Barrels of Import',
                                               line_color = 'purple',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
    
        
        ### Export Plot ###
    
        export_fig = make_lineplot_with_yoybars(filtered_df, main_data= 'total_crude_oil_export',
                                               yoy_data = 'oil_export_perc_change',
                                               title = 'Oil Exports',
                                               yaxis_label= 'Mega Barrels of Export',
                                               line_color = 'darkgreen',
                                               color_scale = 'RdYlGn',
                                               cmin = None, cmax = None)
        
        
        return prod_fig, emp_fig, bar_fig, price_fig, heatmap, import_fig, export_fig, start_date, end_date
    
