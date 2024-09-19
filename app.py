import json
import logging
from typing import Any, Dict, List, Optional
import os

import pandas as pd
import plotly.express as px

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'graph_stube_obj.json')

def load_realonline_data(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """Load and parse JSON data from the REALonline graph_stube_obj file."""
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as file:
            data = json.load(file)
            logging.info(f"Data loaded successfully from {file_path}.")
            return data
    except FileNotFoundError:
        logging.error(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return None

def prepare_treemap_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Prepare data for treemap visualization."""
    treemap_data = []

    for entry in data:
        # Ensure 'e' and 'properties' exist in entry
        if 'e' in entry and 'properties' in entry['e']:
            room_properties = entry['e']['properties']
            room_name = room_properties.get('name', ["Unnamed Room"])
            if isinstance(room_name, list):
                room_name = room_name[0]
            building = room_properties.get('gebaeude', ["Unknown Building"])
            if isinstance(building, list):
                building = building[0]
        else:
            logging.warning(f"Missing 'e' or 'properties' in entry: {entry}")
            continue  # Skip this entry if essential data is missing

        # Process object data if present
        if 'o' in entry and 'properties' in entry['o']:
            obj_properties = entry['o']['properties']
            obj_name = obj_properties.get('bezeichnung', ["Unnamed Object"])
            if isinstance(obj_name, list):
                obj_name = obj_name[0]
            obj_quantity = obj_properties.get('anzahl', [1])
            if isinstance(obj_quantity, list):
                obj_quantity = obj_quantity[0]
            # Assuming 'wert' is a value field; if not present, set to 0
            obj_value = obj_properties.get('wert', [0])
            if isinstance(obj_value, list):
                obj_value = obj_value[0]

            # Ensure quantity is numeric
            try:
                obj_quantity = int(obj_quantity)
            except (ValueError, TypeError):
                obj_quantity = 1

            # Ensure value is numeric
            try:
                obj_value = float(obj_value)
            except (ValueError, TypeError):
                obj_value = 0.0

            treemap_data.append({
                'Building': building,
                'Room': room_name,
                'Object': obj_name,
                'Quantity': obj_quantity,
                'Value': obj_value
            })
        else:
            logging.warning(f"Missing 'o' or 'properties' in entry: {entry}")
            continue  # Skip entries without object data

    # Create a DataFrame
    df = pd.DataFrame(treemap_data)

    if df.empty:
        logging.warning("The DataFrame is empty after processing data.")
        return df

    # Fill missing quantities with 1
    df['Quantity'] = df['Quantity'].fillna(1)

    # Group rare objects under 'Others' to reduce clutter
    object_counts = df['Object'].value_counts()
    threshold = 3  # Objects appearing less than 'threshold' times are grouped
    rare_objects = object_counts[object_counts < threshold].index
    df['Object'] = df['Object'].apply(lambda x: 'Others' if x in rare_objects else x)

    logging.info("Treemap data prepared.")
    return df

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server
app.title = 'Room-Object Treemap Visualization'

# Load data
data = load_realonline_data(DATA_FILE_PATH)
if data:
    df = prepare_treemap_data(data)
    if df.empty:
        logging.error("DataFrame is empty. No data to display.")
        df = pd.DataFrame()
else:
    logging.error("Failed to load data.")
    df = pd.DataFrame()

# Create the app layout
app.layout = html.Div([
    html.H1("Room-Object Treemap Visualization"),

    html.Div([
        html.Div([
            html.Label("Filter by Object Type:"),
            dcc.Dropdown(
                id='object-type-dropdown',
                options=[{'label': obj, 'value': obj} for obj in sorted(df['Object'].unique())] if not df.empty else [],
                multi=True,
                placeholder='Select object types...'
            ),
        ], style={'width': '45%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Search for Room or Object:"),
            dcc.Input(
                id='search-input',
                type='text',
                placeholder='Search...',
                style={'width': '100%'}
            ),
        ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '2%'}),
    ], style={'marginBottom': '20px'}),

    html.Div([
        html.Button('Switch to Sunburst', id='switch-button', n_clicks=0),
    ], style={'marginBottom': '20px'}),

    html.Div([
        dcc.Graph(id='treemap'),
    ]),

    html.Div(id='summary-stats', style={'marginTop': '20px'}),

    # Add an explanatory text
    html.Div([
        html.P("Click on rectangles to zoom in. Hover over them for more info. Use the filters and search to explore the data.")
    ], style={'marginTop': '20px'}),

])

# Define the callback
@app.callback(
    [Output('treemap', 'figure'),
     Output('summary-stats', 'children'),
     Output('switch-button', 'children')],
    [Input('object-type-dropdown', 'value'),
     Input('search-input', 'value'),
     Input('switch-button', 'n_clicks')]
)
def update_figure(selected_objects, search_query, n_clicks):
    # Ensure n_clicks is not None
    if n_clicks is None:
        n_clicks = 0

    # Determine which chart type to display
    chart_type = 'treemap' if n_clicks % 2 == 0 else 'sunburst'
    switch_button_text = 'Switch to Sunburst' if chart_type == 'treemap' else 'Switch to Treemap'

    # Handle case when df is empty
    if df.empty:
        fig = px.imshow([[0]], text_auto=True)
        fig.update_layout(
            title="No data available",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        summary = html.Div([
            html.P("No data to display.")
        ])
        return fig, summary, switch_button_text

    # Filter the dataframe based on inputs
    filtered_df = df.copy()

    if selected_objects:
        filtered_df = filtered_df[filtered_df['Object'].isin(selected_objects)]

    if search_query:
        search_query = search_query.lower()
        filtered_df = filtered_df[filtered_df['Room'].str.lower().str.contains(search_query) |
                                  filtered_df['Object'].str.lower().str.contains(search_query)]

    # Prepare summary statistics
    total_rooms = filtered_df['Room'].nunique()
    total_objects = filtered_df['Object'].nunique()
    total_quantity = filtered_df['Quantity'].sum()

    summary = html.Div([
        html.P(f"Total Rooms: {total_rooms}"),
        html.P(f"Total Objects: {total_objects}"),
        html.P(f"Total Quantity: {total_quantity}")
    ])

    if filtered_df.empty:
        fig = px.imshow([[0]], text_auto=True)
        fig.update_layout(
            title="No data matches the filters",
            xaxis={'visible': False},
            yaxis={'visible': False}
        )
        return fig, summary, switch_button_text

    # Create the treemap or sunburst
    if chart_type == 'treemap':
        fig = px.treemap(
            filtered_df,
            path=['Building', 'Room', 'Object'],
            values='Quantity',
            color='Room',
            hover_data={
                'Quantity': True,
                'Room': True,
                'Object': True,
                'Building': True
            },
            custom_data=['Quantity', 'Building'],
            branchvalues='total'
        )
    else:
        fig = px.sunburst(
            filtered_df,
            path=['Building', 'Room', 'Object'],
            values='Quantity',
            color='Room',
            hover_data={
                'Quantity': True,
                'Room': True,
                'Object': True,
                'Building': True
            },
            custom_data=['Quantity', 'Building']
        )

    # Update hover template
    fig.update_traces(
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Quantity: %{customdata[0]}<br>'
            'Building: %{customdata[1]}<br>'
        ),
        texttemplate='%{label}',
        textposition='middle center'
    )

    # Update layout
    fig.update_layout(
        margin=dict(t=50, l=25, r=25, b=25),
        legend_title='Rooms',
        legend=dict(
            x=1.05,
            y=1,
            traceorder='normal'
        ),
        annotations=[
            dict(
                text="Click on rectangles to zoom in. Hover over them for more info.",
                showarrow=False,
                x=0.5,
                y=-0.1,
                xref="paper",
                yref="paper",
                xanchor='center',
                yanchor='top'
            )
        ]
    )

    return fig, summary, switch_button_text


if __name__ == '__main__':
    app.run_server(debug=False)
