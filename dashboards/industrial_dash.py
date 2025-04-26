#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash
from dash import dcc, html, Input, Output, State, ctx
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


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

def get_data(all_monthly):
    cpi_2016 = all_monthly['cpi_all'][all_monthly.index == '2016-01-31'].values[0]
    ind_df = all_monthly[industry_cols + industry_cols_price_adj]
    ind_df.reset_index(inplace = True)
    ind_df = ind_df[ind_df['date'] >= '2005-01-01']

    #adjust these to 2016 CPI 
    for col in industry_cols_price_adj:
        ind_df[col] = ind_df[col] * (ind_df['cpi_all']/cpi_2016)

    #add total perc changes YoY
    ind_df['total_manufact_orders_perc_change'] = 100 * (ind_df['total_manufact_orders'] - ind_df['total_manufact_orders'].shift(12))/ind_df['total_manufact_orders'].shift(12)
    ind_df['all_output_perc_change'] = 100 * (ind_df['all_output'] - ind_df['all_output'].shift(12))/ind_df['all_output'].shift(12)
    ind_df['manufact_employee_perc_change'] = 100 * (ind_df['manufact_employee'] - ind_df['manufact_employee'].shift(12))/ind_df['manufact_employee'].shift(12)
    ind_df['housing_perc_change'] = 100 * (ind_df['housing_under_construction'] - ind_df['housing_under_construction'].shift(12))/ind_df['housing_under_construction'].shift(12)
    
    return ind_df, cpi_2016

def get_layout(all_monthly):
    
    ind_df, cpi_2016 = get_data(all_monthly)
    
    layout = html.Div([
        html.H2("Industry Dashboard"),

        dcc.DatePickerRange(
            id='date-picker-industry',
            min_date_allowed=ind_df['date'].min(),
            max_date_allowed=ind_df['date'].max(),
        ),
        
        html.Button("Reset", id="reset-button-industry", n_clicks=0, style={'margin': '20px 0'}),
        
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
            html.Div([dcc.Graph(id='heatmap-industry')], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
        ], style={'width': '100%', 'paddingTop': '20px'})
        
    ])
    
    return layout


def register_callbacks(app, all_monthly):

    
    ind_df, cpi_2016 = get_data(all_monthly)

    @app.callback(
        [
         Output('industry_output', 'figure'),
         Output('sub_output', 'figure'),
         Output('manufact_orders', 'figure'),
         Output('sub_orders', 'figure'),
         Output('housing', 'figure'),
         Output('sub_housing', 'figure'),
         Output('heatmap-industry', 'figure'),
         Output('man_emp_fig', 'figure'),
         Output('date-picker-industry', 'start_date'),
         Output('date-picker-industry', 'end_date'),
         Output('subcategory-output-dropdown', 'value'),
         Output('subcategory-orders-dropdown', 'value'),
         Output('subregion-housing-dropdown', 'value')
         ],
        
        [Input('date-picker-industry', 'start_date'),
         Input('date-picker-industry', 'end_date'),
         Input('subcategory-output-dropdown', 'value'),
         Input('subcategory-orders-dropdown', 'value'),
         Input('subregion-housing-dropdown', 'value'),
         Input('reset-button-industry', 'n_clicks')
         
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
    
        if triggered_id == 'reset-button-industry':
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
        
        
        return (industry_output, sub_output, manufact_orders, sub_orders, housing, 
                sub_housing, heatmap, man_emp_fig, start_date, end_date,
                selected_output_subcategories, selected_orders_subcategories, selected_housing_subregions)
    
    
