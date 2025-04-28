#!/usr/bin/env python3


import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from functs import make_lineplot_with_yoybars, generate_basic_heatmap, linreg
import yfinance as yf
import numpy as np



def get_data(all_monthly):
    #get external financial data from yfinance
    start = "1990-01-01"
    yfin_tickers = ["^GSPC","^RUT","GC=F","DX-Y.NYB","USDCNY=X","EURUSD=X"]
    yf_data = yf.download(yfin_tickers, start=start, interval='1d', progress = False)
    yf_data = yf_data['Close']
    yf_data = yf_data.resample('ME').last()
    
    #columns to use from rest of montly data
    monthly_cols = ['cpi_all', 'cpi_energy', 'cpi_food','cpi_shelter', 'cpi_airfare', 
     'cpi_transport', 'cpi_services', 'us_gasprice', 'egg_price', 'bananas_price', 'oil_price', 'electricity_price',
     'oj_price', 'bread_price', 'ground_beef_price', 'tomato_price', 'strawberry_price', 'coffee_price',
     'all_construction_spend', 'housing_under_construction', 'total_manufact_orders', 'durable_manufact_orders', 
     'machinery_manufact_orders', 'consumer_goods_manufact_orders', 'vehicle_manufact_orders', 
     'construction_materials_manufact_orders', 'population', 'federal_outlays', 
     'all_imports', 'all_exports', 
     'unemployment', 'computer_manufact_orders', 'furnature_manufact_orders', 'capital_manufact_orders', 
     'transport_manufact_orders', 'it_manufact_orders', 
     
     # Outputs are quarterly which messes up rolling calcs
     #'transport_output', 'other_output', 'all_output', 
     #'manufact_output', 'construction_output', 'buisness_service_output', 'retail_output', 'agriculture_output', 
     #'finance_output', 'mining_output', 'it_output', 'utilities_output', 'gov_output', 
     
     'neast_housing_under_construction', 'west_housing_under_construction', 'south_housing_under_construction', 
     'mwest_housing_under_construction', 'job_opening','layoffs','all_employees', 'manufact_employee', 
     'oil_employees','federal_employees',  'health_employees', 'buisness_employees', 
     'agriculture_employees', 'mining_employees', 'wholesale_employees', 'truck_employees', 
     'it_employees', 'retail_employees', 'leisure_employees', 'finance_employees', 'construction_employees', 
     'participation_rate', 'participation_25_54_rate', 'participation_55+_rate', 
      'participation_women_rate', 'participation_men_rate', 
     'participation_immigrant_rate','us_natural_gas_prod', 'total_crude_oil_prod', 
     'total_crude_oil_import', 'total_crude_oil_export', 
     'total_million_tons_co2']
    
    col_renames = new_names = [
        'CPI All', 'CPI Energy', 'CPI Food', 'CPI Shelter', 'CPI Airfare',
        'CPI Transport', 'CPI Services', 'Gas Price', 'Egg Price', 'Banana Price', 'Oil Price', 'Electricity Price',
        'OJ Price', 'Bread Price', 'Beef Price', 'Tomato Price', 'Strawberry Price', 'Coffee Price',
        'Construction Spend', 'Housing Build', 'Manufacturing Orders', 'Durable Orders',
        'Machinery Orders', 'Goods Orders', 'Vehicle Orders', 'Construction Materials',
        'Population', 'Federal Spending',
        'Imports', 'Exports',
        'Unemployment', 'Computer Orders', 'Furniture Orders', 'Capital Orders',
        'Transport Orders', 'IT Orders', 
        
        #'Transport Output', 'Other Output', 'Total Output',
        #'Manufacturing Output', 'Construction Output', 'Business Services', 'Retail Output', 'Agriculture Output',
        #'Finance Output', 'Mining Output', 'IT Output', 'Utilities Output', 'Government Output',
        
        'Northeast Housing', 'West Housing', 'South Housing',
        'Midwest Housing', 'Job Openings', 'Layoffs', 'Total Employees', 'Manufacturing Jobs',
        'Oil Jobs', 'Federal Jobs', 'Health Jobs', 'Business Jobs',
        'Agriculture Jobs', 'Mining Jobs', 'Wholesale Jobs', 'Truck Jobs',
        'IT Jobs', 'Retail Jobs', 'Leisure Jobs', 'Finance Jobs', 'Construction Jobs',
        'Participation Rate', 'Participation Rate (24-55)', 'Participation Rate (55+)', 
        'Participation Rate (Women)', 'Participation Rate (Men)',
        'Participation Rate (Immigrant)', 'Natural Gas Output', 'Crude Oil Output',
        'Crude Oil Import', 'Crude Oil Export',
        'CO2 Emissions'
    ]

    monthly_data = all_monthly[monthly_cols]
    monthly_data.columns = col_renames
    monthly_data = pd.merge(yf_data, monthly_data, left_index=True, right_index=True)
    monthly_data.columns = ['SP500', 'Rut2000', 'Gold', 'DXY', 'CHY', 'Euro'] + col_renames

    corr_df = monthly_data.copy()
    corr_df.reset_index(inplace = True)
    corr_df.columns = ['date'] + ['SP500', 'Rut2000', 'Gold', 'DXY', 'CHY', 'Euro'] + col_renames
    
    return corr_df, col_renames

def get_layout(all_monthly):
    
    corr_df, _ = get_data(all_monthly)

    layout = html.Div([
        html.H2("Correlations Dashboard"),
        
        html.Button("Reset", id="reset-button-corr", n_clicks=0, style={'margin': '20px 0'}),
    
        dcc.DatePickerRange(
            id='date-picker-corr',
            min_date_allowed=corr_df['date'].min(),
            max_date_allowed=corr_df['date'].max(),
            start_date=corr_df['date'].min(),
            end_date=corr_df['date'].max()
        ),
    
        html.Div([
            html.Div([dcc.Graph(id='corr-heatmap')], style={'width': '100%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
    
        html.Div([
            html.Div([
                dcc.Graph(id='rolling-corr'),
                html.Div(
                    "Correlations are the linear correlation coefficient describing the direction and strength of the relationship between the variables. Beta values decribe the magnitude of the relationship. Beta values can be interpreted as 'a 1% increase in one variable results in an N% change in the other.'",
                    style={
                        'textAlign': 'center',
                        'marginTop': '10px',
                        'fontSize': '16px',
                        'color': 'gray'
                    }
                )
            ], style={'width': '100%', 'display': 'inline-block'}),
        ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),

    
        
    ])

    return layout


def register_callbacks(app, all_monthly):
    
    corr_df, col_renames = get_data(all_monthly)
    
    def update_lineplot(clickData, filtered_df, window = 12):
        if clickData is None:
            return go.Figure()
    
        # Parse clicked features
        feature1 = clickData['points'][0]['x']
        feature2 = clickData['points'][0]['y']
        
        rolling_corr = filtered_df[feature1].rolling(window=window).corr(filtered_df[feature2])
        
        filtered_df_log = np.log(filtered_df.iloc[:, 1:])
        filtered_df_log['date'] = filtered_df['date']
        rolling_beta = linreg(filtered_df_log[feature1], filtered_df_log[feature2])
        
    
        fig = go.Figure()
    
        fig.add_trace(go.Scatter(
                x=filtered_df['date'],
                y=rolling_corr,
                name='Rolling Correlation (12mo)',
                yaxis='y1',
                line=dict(color='blue')
            ))
    
        fig.add_trace(go.Scatter(
                x=filtered_df['date'],
                y=rolling_beta,
                name='Rolling Beta (12mo)',
                yaxis='y2',
                line=dict(color='orange')
            ))
    
        fig.update_layout(
            title=f"Rolling Correlation and Beta: {feature1} vs {feature2}",
            xaxis=dict(title='Date'),
            yaxis=dict(
                title='Rolling Pearson Correlation',
                side='left',
                range=[-1, 1]
            ),
            yaxis2=dict(
                title='Rolling Beta',
                overlaying='y',
                side='right'
            ),
            width=1000,
            height=500,
            legend=dict(
                orientation="h",  # Horizontal legend
                y=-0.25,  # Position it above the plot
                x=0.5,  # Center horizontally
                xanchor="center",  # Align legend to the center
                yanchor="bottom"   # Align to the bottom of the legend box
            )
        )
    
        return fig

    @app.callback(
        [Output('corr-heatmap', 'figure'),
         Output('rolling-corr', 'figure'),
         
         Output('date-picker-corr', 'start_date'),
         Output('date-picker-corr', 'end_date'),],
        [Input('date-picker-corr', 'start_date'),
         Input('date-picker-corr', 'end_date'),
         Input('reset-button-corr', 'n_clicks'),
         Input('corr-heatmap', 'clickData')]
    )

    def update_graphs(start_date, end_date, reset_clicks, clickData):
        from dash import  ctx
        triggered_id = ctx.triggered_id
        
        # Set default values
        default_start = corr_df['date'].min()
        default_end = corr_df['date'].max()
        if start_date is None:
            start_date = default_start
        if end_date is None:
            end_date = default_end
    
        if triggered_id == 'reset-button-corr':
            start_date = default_start
            end_date = default_end
        
        
        # Filter data
        mask = (corr_df['date'] >= start_date) & (corr_df['date'] <= end_date)
        filtered_df = corr_df.loc[mask]
        
        heatmap = generate_basic_heatmap(filtered_df, 
                                         ['SP500', 'Rut2000', 'Gold', 'DXY', 'CHY', 'Euro'] + col_renames,
                                         ['SP500', 'Rut2000', 'Gold', 'DXY', 'CHY', 'Euro'] + col_renames,
                                         width=900, height=900, title = False)
        
        corr_fig = update_lineplot(clickData, filtered_df)
    
        return heatmap, corr_fig, start_date, end_date



