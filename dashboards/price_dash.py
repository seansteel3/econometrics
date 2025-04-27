#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def make_lineplot_with_yoybars(df, main_data, yoy_data, title, yaxis_label, color_scale, line_color = 'royalblue', cmin = None, cmax = None):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],  # 70% line plot, 30% bar plot
        vertical_spacing=0.02,
        subplot_titles=(f"{title}", " ")
    )
    

    #production
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df[main_data],
            name=f"{title}",
            mode='lines',
            line=dict(color=line_color)
        ),
        row=1, col=1
    )
    #YoY Change
    bar_colors = df[yoy_data]
    
    if cmin is None:
        cmin = bar_colors.min()
        cmax = bar_colors.max()
    else:
        cmin = cmin
        cmax = cmax
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df[yoy_data],
            marker=dict(
                color=bar_colors,
                colorscale=color_scale,  # red-yellow-green
                cmin=cmin ,
                cmax=cmax,
                colorbar=dict(title="YoY Change"),
                showscale = False
            ),
        ),
        row=2, col=1
    )
    fig.update_layout(
        height=500,
        showlegend=False,
        title=f"{title} with YoY Change",
        xaxis2=dict(title="Date"),  # applies to shared x-axis
        yaxis=dict(title=yaxis_label),
        yaxis2=dict(title="YoY Change (%)"),
    )
    
    return fig


def get_data(all_monthly):
    cpi_cols = ['cpi_all', 'cpi_energy', 'cpi_food', 'cpi_used_car', 'cpi_shelter', 
                'cpi_new_car', 'cpi_airfare', 'cpi_transport', 'cpi_services'
                
                ]
    
    rate_cols = ['inflation_rate'
                 ]
    
    price_cols = ['egg_price', 'oj_price', 'bread_price', 'ground_beef_price', 
                  #'chicken_price',  #starts 2006
                  'tomato_price', 'strawberry_price', 'coffee_price', 'bananas_price'
                  ]


    price_df = all_monthly.copy()
    price_df = price_df[cpi_cols + rate_cols + price_cols]
    price_df.reset_index(inplace=True)
    price_df = price_df[price_df['date'] >= '1995-01-01']
    
    price_df['coffee_price'] = np.round(price_df['coffee_price']/100, 2)#convert from cents to dollars
    # Derived perc changes
    for col in price_cols:
        price_df[f'{col}_perc_change'] = 100 * (price_df[col] - price_df[col].shift(12))/ price_df[col].shift(12)
    
    for col in cpi_cols:
        price_df[f'{col}_perc_change'] = 100 * (price_df[col] - price_df[col].shift(12))/ price_df[col].shift(12)
    
    
    
    return price_df


def get_layout(all_monthly):
    
    price_df = get_data(all_monthly)


    layout = html.Div([
        html.H2("Prices Dashboard"),
    
        dcc.DatePickerRange(
            id='date-picker-prices',
            min_date_allowed=price_df['date'].min(),
            max_date_allowed=price_df['date'].max(),
        ),
        
        html.Button("Reset", id="reset-button-prices", n_clicks=0, style={'margin': '20px 0'}),
        
        html.Div([
            html.Div([dcc.Graph(id='inflation-fig')], style={'width': '100%', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
    
         
        html.Div([
            html.Div([dcc.Graph(id='energycpi-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='foodcpi-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='airfarecpi-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='transportcpi-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
    
         html.Div([
             html.Div([dcc.Graph(id='sheltercpi-fig')], style={'width': '48%', 'display': 'inline-block'}),
             html.Div([dcc.Graph(id='servicescpi-fig')], style={'width': '48%', 'display': 'inline-block'}),
         ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='usedcarcpi-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='newcarcpi-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='egg-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='oj-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='bread-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='beef-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        html.Div([
            html.Div([dcc.Graph(id='coffee-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='bananas-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        
        html.Div([
            html.Div([dcc.Graph(id='tomato-fig')], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='strawberry-fig')], style={'width': '48%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'paddingTop': '20px'}),
        
        
        
    ])
    return layout

def register_callbacks(app, all_monthly):
    
    price_df = get_data(all_monthly)
    @app.callback(
        [
         Output('inflation-fig', 'figure'),
         Output('egg-fig', 'figure'),
         Output('oj-fig', 'figure'),
         Output('bread-fig', 'figure'),
         Output('beef-fig', 'figure'),
         Output('coffee-fig', 'figure'),
         Output('bananas-fig', 'figure'),
         Output('tomato-fig', 'figure'),
         Output('strawberry-fig', 'figure'),
         
         Output('energycpi-fig', 'figure'),
         Output('foodcpi-fig', 'figure'),
         Output('usedcarcpi-fig', 'figure'),
         Output('newcarcpi-fig', 'figure'),
         Output('sheltercpi-fig', 'figure'),
         Output('servicescpi-fig', 'figure'),
         Output('airfarecpi-fig', 'figure'),
         Output('transportcpi-fig', 'figure'),
         
         
         Output('date-picker-prices', 'start_date'),
         Output('date-picker-prices', 'end_date'),
         ],
        
        [Input('date-picker-prices', 'start_date'),
         Input('date-picker-prices', 'end_date'),
         Input('reset-button-prices', 'n_clicks')
         
         ]
    )
    def update_graphs(start_date, end_date, reset_clicks):
        #set defaults for reset
        triggered_id = ctx.triggered_id
        
        # Set default values
        default_start = price_df['date'].min()
        default_end = price_df['date'].max()
        if start_date is None:
            start_date = default_start
        if end_date is None:
            end_date = default_end
    
        if triggered_id == 'reset-button-prices':
            start_date = default_start
            end_date = default_end
            
        
        # Filter data
        mask = (price_df['date'] >= start_date) & (price_df['date'] <= end_date)
        filtered_df = price_df.loc[mask]
        date_col = filtered_df['date']
    
        
        ### Inflation Figure ###
        inflation_fig = go.Figure()
        inflation_fig.add_trace(
            go.Scatter(x=date_col, y=filtered_df['inflation_rate'], 
                       name='Inflation Rate',
                       line=dict(color='royalblue')))
    
        inflation_fig.update_layout(
            height=300,
            margin=dict(l=50, r=10, t=40, b=40),
            title="Inflation Rate",
            xaxis_title="Date",
            yaxis_title="Rate",
            hovermode="x unified"
        )
        
        
        ### Price sets ###
        eggs_price = make_lineplot_with_yoybars(filtered_df, main_data= 'egg_price',
                                               yoy_data = 'egg_price_perc_change',
                                               title = 'Egg Price',
                                               yaxis_label= 'Price ($/dozen)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        oj_price = make_lineplot_with_yoybars(filtered_df, main_data= 'oj_price',
                                               yoy_data = 'oj_price_perc_change',
                                               title = 'Orange Juice Price',
                                               yaxis_label= 'Price ($/12oz)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        bread_price = make_lineplot_with_yoybars(filtered_df, main_data= 'bread_price',
                                               yoy_data = 'bread_price_perc_change',
                                               title = 'Bread Price',
                                               yaxis_label= 'Price ($/lb)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        ground_beef_price = make_lineplot_with_yoybars(filtered_df, main_data= 'ground_beef_price',
                                               yoy_data = 'ground_beef_price_perc_change',
                                               title = 'Ground Beef Price',
                                               yaxis_label= 'Price ($/lb)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        coffee_price = make_lineplot_with_yoybars(filtered_df, main_data= 'coffee_price',
                                               yoy_data = 'coffee_price_perc_change',
                                               title = 'Coffee Price',
                                               yaxis_label= 'Price ($/lb)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        bananas_price = make_lineplot_with_yoybars(filtered_df, main_data= 'bananas_price',
                                               yoy_data = 'bananas_price_perc_change',
                                               title = 'Bananas Price',
                                               yaxis_label= 'Price ($/lb)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        tomato_price = make_lineplot_with_yoybars(filtered_df, main_data= 'tomato_price',
                                               yoy_data = 'tomato_price_perc_change',
                                               title = 'Tomato Price',
                                               yaxis_label= 'Price ($/lb)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        strawberry_price = make_lineplot_with_yoybars(filtered_df, main_data= 'strawberry_price',
                                               yoy_data = 'strawberry_price_perc_change',
                                               title = 'Strawberry Price',
                                               yaxis_label= 'Price ($/lb)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        ### CPI sets ###
        cpi_energy = make_lineplot_with_yoybars(filtered_df, main_data= 'cpi_energy',
                                               yoy_data = 'cpi_energy_perc_change',
                                               title = 'Energy CPI',
                                               yaxis_label= 'CPI (1982)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        cpi_food = make_lineplot_with_yoybars(filtered_df, main_data= 'cpi_food',
                                               yoy_data = 'cpi_food_perc_change',
                                               title = 'Food CPI',
                                               yaxis_label= 'CPI (1982)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        cpi_used_car = make_lineplot_with_yoybars(filtered_df, main_data= 'cpi_used_car',
                                               yoy_data = 'cpi_used_car_perc_change',
                                               title = 'Used Vehicle CPI',
                                               yaxis_label= 'CPI (1982)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        cpi_new_car = make_lineplot_with_yoybars(filtered_df, main_data= 'cpi_new_car',
                                               yoy_data = 'cpi_new_car_perc_change',
                                               title = 'New Vehicle CPI',
                                               yaxis_label= 'CPI (1982)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        cpi_shelter = make_lineplot_with_yoybars(filtered_df, main_data= 'cpi_shelter',
                                               yoy_data = 'cpi_shelter_perc_change',
                                               title = 'Shelter CPI',
                                               yaxis_label= 'CPI (1982)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        cpi_services = make_lineplot_with_yoybars(filtered_df, main_data= 'cpi_services',
                                               yoy_data = 'cpi_services_perc_change',
                                               title = 'Services CPI',
                                               yaxis_label= 'CPI (1982)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        cpi_airfare = make_lineplot_with_yoybars(filtered_df, main_data= 'cpi_airfare',
                                               yoy_data = 'cpi_airfare_perc_change',
                                               title = 'Airfare CPI',
                                               yaxis_label= 'CPI (1982)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        cpi_transport = make_lineplot_with_yoybars(filtered_df, main_data= 'cpi_transport',
                                               yoy_data = 'cpi_transport_perc_change',
                                               title = 'City Transport CPI',
                                               yaxis_label= 'CPI (1982)',
                                               color_scale = 'RdYlGn_r',
                                               cmin = None, cmax = None)
        
        
        return (inflation_fig, eggs_price, oj_price, bread_price, ground_beef_price, 
                coffee_price, bananas_price,
                tomato_price, strawberry_price, cpi_energy, cpi_food, 
                cpi_used_car, cpi_new_car, cpi_shelter, cpi_services, cpi_airfare, 
                cpi_transport, start_date, end_date)
    

