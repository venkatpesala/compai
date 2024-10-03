import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# -------------------------------
# 1. Setup Logging
# -------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# 2. Data Cleaning and Preprocessing
# -------------------------------

def clean_monetary_columns(df, columns):
    """
    Cleans monetary columns by removing symbols, handling negative values, and converting to float.
    """
    for col in columns:
        # Remove dollar signs, commas, and handle negative values
        df[col] = df[col].astype(str).replace({
            r'\$': '',
            ',': '',
            r'\(': '-',
            r'\)': ''
        }, regex=True)
        
        # Handle suffixes like M and B
        df[col] = df[col].replace({
            'M': 'e6',
            'B': 'e9',
            'K': 'e3'
        }, regex=True)
        
        # Convert to float, coercing errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

# Load the dataset
file_path = 'CompAI Peer Group Fundamentals 20240928 MSFT v.2.xlsx'  # Update with your file path
df = pd.read_excel(file_path)

# Display initial data
logger.info("Initial data loaded:")
logger.info(df.head())

# Columns that contain monetary values
monetary_columns = [
    'Salary (USD)', 'Bonus (USD)', 'Stock Awards (USD)',
    'Non-Equity Incentive Plan Compensation (USD)', 'All Other Compensation (USD)',
    'Total CEO Compensation (Check)', 'CEO Total Compensation (2023)'
]

# Clean monetary columns
df = clean_monetary_columns(df, monetary_columns)

# Display data after cleaning monetary columns
logger.info("Data after cleaning monetary columns:")
logger.info(df[monetary_columns].head())

# Handle missing values using KNN Imputer
imputer = KNNImputer(n_neighbors=5)
try:
    imputed_data = imputer.fit_transform(df[monetary_columns])
    logger.info(f"Imputed data shape: {imputed_data.shape}")
    
    # Ensure that imputed_data has the same number of columns as monetary_columns
    if imputed_data.shape[1] != len(monetary_columns):
        raise ValueError("Imputed data columns do not match monetary_columns.")
    
    df[monetary_columns] = imputed_data
except ValueError as ve:
    logger.error(f"ValueError during imputation: {ve}")
except Exception as e:
    logger.error(f"Unexpected error during imputation: {e}")

# Replace any remaining missing values with 0 (if any)
df[monetary_columns] = df[monetary_columns].fillna(0)

# Display data after imputation
logger.info("Data after KNN imputation:")
logger.info(df[monetary_columns].head())

# Additional numerical columns for clustering (performance metrics)
performance_metrics = [
    'Market Cap (USD Billion)', 'Revenue FY 2023 (USD Billion)', 'EPS', 
    'P/E Ratio', 'ROE (%)', 'Free Cash Flow (USD Billion)', 
    'Dividend Yield (%)', '1-Year Stock Performance (%)', 
    '5-Year Stock Performance (%)', 'Total Revenue Growth (%)', 
    'Profit Margin (%)', 'Debt-to-Equity Ratio'
]

# Convert performance metrics to numeric (handle any non-numeric entries)
for col in performance_metrics:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing performance metrics with median values
df[performance_metrics] = df[performance_metrics].fillna(df[performance_metrics].median())

# Display data after handling performance metrics
logger.info("Data after cleaning performance metrics:")
logger.info(df[performance_metrics].head())

# Extract peer group tickers and convert to dummy variables
df['Peer Group Company Tickers'] = df['Peer Group Company Tickers'].fillna('')
peer_dummies = df['Peer Group Company Tickers'].str.strip().str.get_dummies(sep=',').astype(int)

# Combine peer group dummies with the main dataframe
df = pd.concat([df, peer_dummies], axis=1)

# Display final preprocessed data
logger.info("Final preprocessed data:")
logger.info(df.head())

# Verify data types to ensure no object types are present in numerical columns
logger.info("Data types after preprocessing:")
logger.info(df.dtypes)

# Identify columns with object dtype
object_cols = df.select_dtypes(include=['object']).columns.tolist()
logger.info(f"Columns with object dtype: {object_cols}")

# Determine numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
logger.info(f"Numeric columns: {numeric_columns}")

# -------------------------------
# 3. Initialize Dash App with Bootstrap
# -------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deploying to platforms like Heroku

# -------------------------------
# 4. Define Cluster Naming Function
# -------------------------------

def generate_cluster_names(cluster_type, k):
    """
    Generates descriptive names for clusters based on their centroids.
    Ensures each cluster has a unique and descriptive name by considering multiple features.
    """
    names = []
    centroids = cluster_centroids.get(cluster_type, {}).get(k, pd.DataFrame())
    
    if centroids.empty:
        # If centroids are not precomputed, return generic names
        return [f"Cluster {i+1}" for i in range(k)]
    
    for index, row in centroids.iterrows():
        name = f"Cluster {index + 1}"
        
        if cluster_type == 'performance':
            # Example logic based on Market Cap and Revenue
            if row['Market Cap (USD Billion)'] > df['Market Cap (USD Billion)'].median():
                name = "Large Cap"
            else:
                name = "Small Cap"
            if row['Revenue FY 2023 (USD Billion)'] > df['Revenue FY 2023 (USD Billion)'].median():
                name += " High Revenue"
            else:
                name += " Low Revenue"
        
        elif cluster_type == 'peer':
            # Example logic based on number of peer overlaps
            num_peers = row.sum()
            if num_peers > len(peer_dummies.columns) * 0.5:
                name = "High Peer Overlap"
            else:
                name = "Low Peer Overlap"
        
        elif cluster_type == 'compensation':
            # Example logic based on Salary and Bonus
            if row['Salary (USD)'] > df['Salary (USD)'].median():
                name = "High Salary"
            else:
                name = "Moderate Salary"
            if row['Bonus (USD)'] > df['Bonus (USD)'].median():
                name += " & High Bonus"
            else:
                name += " & Moderate Bonus"
        
        elif cluster_type == 'individual':
            # Example logic based on Bonus and Stock Awards
            if row['Bonus (USD)'] > df['Bonus (USD)'].median():
                name = "High Bonus"
            else:
                name = "Moderate Bonus"
            if row['Stock Awards (USD)'] > df['Stock Awards (USD)'].median():
                name += " & High Stock Awards"
            else:
                name += " & Moderate Stock Awards"
        
        # Append cluster index to ensure uniqueness
        name += f" ({index + 1})"
        
        names.append(name)
    
    return names

# -------------------------------
# 5. Define Dash Layout
# -------------------------------

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Company Clustering Dashboard"), width=12)
    ], justify='center', className='my-4'),
    
    dbc.Row([
        dbc.Col([
            html.Label('Select Clustering Type:'),
            dcc.Dropdown(
                id='cluster-type',
                options=[
                    {'label': 'Performance Metrics', 'value': 'performance'},
                    {'label': 'Peer Group Tickers', 'value': 'peer'},
                    {'label': 'CEO Compensation Style', 'value': 'compensation'},
                    {'label': 'Individual Compensation Components', 'value': 'individual'}
                ],
                value='performance',
                clearable=False
            ),
        ], md=3),
        
        dbc.Col([
            html.Label('Select Number of Clusters (k):'),
            dcc.Dropdown(
                id='num-clusters',
                options=[{'label': str(k), 'value': k} for k in range(2, 11)],
                value=5,
                clearable=False
            ),
        ], md=3),
        
        dbc.Col([
            html.Label('Select Distance Metric:'),
            dcc.Dropdown(
                id='distance-metric',
                options=[
                    {'label': 'Euclidean (L2)', 'value': 'euclidean'},
                    {'label': 'Manhattan (L1)', 'value': 'manhattan'},
                    {'label': 'Cosine', 'value': 'cosine'}
                ],
                value='euclidean',
                clearable=False
            ),
        ], md=3),
        
        dbc.Col([
            html.Label('Select Linkage Method:'),
            dcc.Dropdown(
                id='linkage-method',
                options=[
                    {'label': 'Average', 'value': 'average'},
                    {'label': 'Complete', 'value': 'complete'},
                    {'label': 'Single', 'value': 'single'}
                ],
                value='average',
                clearable=False
            ),
        ], md=3),
    ], className='mb-4'),
    
    dbc.Row([
        dbc.Col([
            html.Label('Filter by Industry:'),
            dcc.Dropdown(
                id='filter-industry',
                options=[{'label': industry, 'value': industry} for industry in sorted(df['Industry'].dropna().unique())],
                value=[],
                multi=True,
                placeholder="Select industries"
            ),
        ], md=3),
        
        dbc.Col([
            html.Label('Filter by Sector:'),
            dcc.Dropdown(
                id='filter-sector',
                options=[{'label': sector, 'value': sector} for sector in sorted(df['Sector'].dropna().unique())],
                value=[],
                multi=True,
                placeholder="Select sectors"
            ),
        ], md=3),
        
        dbc.Col([
            html.Label(''),
            html.Button("Download Clustered Data", id="download-button", className="btn btn-primary", style={'width': '100%'}),
            dcc.Download(id="download-dataframe-csv"),
        ], md=3),
        
        dbc.Col([
            html.Label(''),
            # Placeholder for alignment
        ], md=3),
    ], className='mb-4'),
    
    dbc.Row([
        dbc.Col([
            html.H3("Cluster Statistics"),
            dbc.Table(id='cluster-stats', bordered=True, striped=True, hover=True, responsive=True)
        ], md=6),
        
        dbc.Col([
            html.H3("Correlation Heatmap"),
            dcc.Graph(id='correlation-heatmap'),
        ], md=6),
    ], className='mb-4'),
    
    dbc.Row([
        dbc.Col([
            html.H3("PCA-based Clustering Visualization"),
            dcc.Graph(id='pca-graph'),
        ], md=6),
        
        dbc.Col([
            html.H3("Cluster Scatter Plot"),
            dcc.Graph(id='cluster-graph'),
        ], md=6),
    ], className='mb-4'),
    
    dbc.Row([
        dbc.Col([
            html.H3("Help & Instructions"),
            html.Div([
                html.P("1. **Select Clustering Type**: Choose the metric based on which companies will be clustered."),
                html.P("2. **Select Number of Clusters (k)**: Define how many clusters you want to form."),
                html.P("3. **Select Distance Metric**: Choose the distance metric (similarity measure) for clustering."),
                html.P("4. **Select Linkage Method**: Choose the linkage method for Agglomerative Clustering."),
                html.P("5. **Filter by Industry and Sector**: Narrow down the companies displayed based on industry or sector."),
                html.P("6. **Download Clustered Data**: Click the button to download the current clustering results as a CSV file."),
                html.P("7. **Cluster Statistics**: View mean values of each feature per cluster to understand cluster characteristics."),
                html.P("8. **Correlation Heatmap**: Analyze the correlation between different numerical features."),
                html.P("9. **PCA-based Visualization**: View a dimensionality-reduced visualization of clusters."),
                html.P("10. **Cluster Scatter Plot**: Visualize clusters based on selected features.")
            ], style={'textAlign': 'left'})
        ], md=12)
    ])
    
], fluid=True)

# -------------------------------
# 6. Define Callbacks
# -------------------------------

@app.callback(
    [
        Output('cluster-graph', 'figure'),
        Output('cluster-stats', 'children'),
        Output('correlation-heatmap', 'figure'),
        Output('pca-graph', 'figure')
    ],
    [
        Input('cluster-type', 'value'),
        Input('num-clusters', 'value'),
        Input('distance-metric', 'value'),
        Input('linkage-method', 'value'),
        Input('filter-industry', 'value'),
        Input('filter-sector', 'value')
    ]
)
def update_output(cluster_type, num_clusters, distance_metric, linkage_method, selected_industries, selected_sectors):
    logger.info(f"Selected cluster type: {cluster_type}, k: {num_clusters}, distance_metric: {distance_metric}, linkage: {linkage_method}")
    logger.info(f"Selected industries: {selected_industries}")
    logger.info(f"Selected sectors: {selected_sectors}")
    
    # Apply filters
    filtered_df = df.copy()
    if selected_industries:
        filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industries)]
    if selected_sectors:
        filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sectors)]
    
    logger.info(f"Number of records after filtering: {len(filtered_df)}")
    
    if filtered_df.empty:
        logger.warning("No data after applying filters.")
        # Return empty figures and a warning message
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No data available for the selected filters.")
        empty_table = dbc.Alert("No data available for the selected filters.", color="warning")
        return empty_fig, empty_table, empty_fig, empty_fig
    
    # Define relevant columns for clustering based on cluster_type
    clustering_features = []
    if cluster_type == 'performance':
        clustering_features = performance_metrics
    elif cluster_type == 'peer':
        clustering_features = peer_dummies.columns.tolist()
    elif cluster_type == 'compensation':
        clustering_features = [
            'Salary (USD)', 'Bonus (USD)', 'Stock Awards (USD)',
            'Non-Equity Incentive Plan Compensation (USD)', 
            'All Other Compensation (USD)', 'Total CEO Compensation (Check)'
        ]
    elif cluster_type == 'individual':
        clustering_features = [
            'Salary (USD)', 'Bonus (USD)', 'Stock Awards (USD)',
            'Non-Equity Incentive Plan Compensation (USD)', 
            'All Other Compensation (USD)'
        ]
    else:
        clustering_features = performance_metrics  # Default
    
    # Ensure there are enough features and samples for clustering
    if len(clustering_features) < 1:
        logger.warning("No clustering features available.")
        empty_fig = go.Figure()
        empty_fig.update_layout(title="No clustering features available.")
        empty_table = dbc.Alert("No clustering features available.", color="warning")
        return empty_fig, empty_table, empty_fig, empty_fig
    
    # Prepare data for clustering
    clustering_data = filtered_df[clustering_features]
    
    # Handle missing values by filling with median
    clustering_data = clustering_data.fillna(clustering_data.median())
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    # Perform Agglomerative Clustering
    try:
        clustering_model = AgglomerativeClustering(
            n_clusters=num_clusters,
            metric=distance_metric,
            linkage=linkage_method
        )
        cluster_labels = clustering_model.fit_predict(scaled_data)
        logger.info("Clustering completed successfully.")
    except ValueError as ve:
        logger.error(f"ValueError during clustering: {ve}")
        # Return empty figures and an error message
        empty_fig = go.Figure()
        empty_fig.update_layout(title=f"Clustering failed: {ve}")
        empty_table = dbc.Alert(f"Clustering failed: {ve}", color="danger")
        return empty_fig, empty_table, empty_fig, empty_fig
    except Exception as e:
        logger.error(f"Unexpected error during clustering: {e}")
        # Return empty figures and an error message
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Clustering failed due to an unexpected error.")
        empty_table = dbc.Alert("Clustering failed due to an unexpected error.", color="danger")
        return empty_fig, empty_table, empty_fig, empty_fig
    
    # Assign cluster labels to the filtered DataFrame
    filtered_df['Cluster'] = cluster_labels  # Direct assignment without .values
    
    # Generate cluster names
    # For naming, compute the mean of features for each cluster to simulate centroids
    cluster_centroids_sim = filtered_df.groupby('Cluster')[clustering_features].mean().reset_index()
    cluster_centroids_sim['Cluster'] = cluster_centroids_sim['Cluster'].astype(int)
    
    # Create a temporary dictionary to store centroids for naming
    global cluster_centroids
    cluster_centroids = {}
    cluster_centroids[cluster_type] = {}
    cluster_centroids[cluster_type][num_clusters] = cluster_centroids_sim.set_index('Cluster')
    
    cluster_names = generate_cluster_names(cluster_type, num_clusters)
    
    # Define relevant columns for statistics (exclude peer tickers and binary columns)
    peer_ticker_cols = peer_dummies.columns.tolist()
    stats_columns = [col for col in numeric_columns if col not in peer_ticker_cols]
    
    # Calculate descriptive statistics for each cluster
    cluster_stats = filtered_df.groupby('Cluster')[stats_columns].mean().reset_index()
    cluster_stats['Cluster Name'] = cluster_stats['Cluster'].apply(lambda x: cluster_names[x] if x < len(cluster_names) else f'Cluster {x + 1}')
    
    # Prepare cluster statistics table using Dash Bootstrap Components
    # Rearrange columns to show 'Cluster' and 'Cluster Name' first
    cluster_stats = cluster_stats[['Cluster', 'Cluster Name'] + [col for col in cluster_stats.columns if col not in ['Cluster', 'Cluster Name']]]
    
    # Convert to Dash Table
    table_header = [
        html.Thead(html.Tr([html.Th("Cluster"), html.Th("Cluster Name")] + [html.Th(col) for col in cluster_stats.columns if col not in ['Cluster', 'Cluster Name']]))
    ]
    
    table_body = [
        html.Tr([
            html.Td(row['Cluster']),
            html.Td(row['Cluster Name']),
        ] + [
            html.Td(f"{row[col]:,.2f}") for col in cluster_stats.columns if col not in ['Cluster', 'Cluster Name']
        ]) for index, row in cluster_stats.iterrows()
    ]
    
    cluster_stats_html = dbc.Table(table_header + [html.Tbody(table_body)], bordered=True, striped=True, hover=True, responsive=True)
    
    # Determine plot features based on cluster type
    if cluster_type == 'performance':
        x_feature = 'Market Cap (USD Billion)'
        y_feature = 'Revenue FY 2023 (USD Billion)'
        title = 'Performance Metrics Clustering'
    elif cluster_type == 'peer':
        x_feature = 'Market Cap (USD Billion)'
        y_feature = 'Revenue FY 2023 (USD Billion)'
        title = 'Peer Group Tickers Clustering'
    elif cluster_type == 'compensation':
        x_feature = 'Salary (USD)'
        y_feature = 'Total CEO Compensation (Check)'
        title = 'CEO Compensation Style Clustering'
    elif cluster_type == 'individual':
        x_feature = 'Bonus (USD)'
        y_feature = 'Stock Awards (USD)'
        title = 'Individual Compensation Components Clustering'
    else:
        x_feature = 'Market Cap (USD Billion)'
        y_feature = 'Revenue FY 2023 (USD Billion)'
        title = 'Clustering'
    
    # Create scatter plot
    scatter_fig = px.scatter(
        filtered_df,
        x=x_feature,
        y=y_feature,
        color='Cluster',
        title=title,
        hover_name='Company Name',
        size='Market Cap (USD Billion)',  # Bubble size based on Market Cap
        template='plotly_white',
        labels={'Cluster': 'Cluster'}
    )
    
    # Update cluster scatter plot layout
    scatter_fig.update_layout(
        xaxis_title=x_feature,
        yaxis_title=y_feature,
        legend_title_text='Cluster',
        legend=dict(
            itemsizing='constant'
        )
    )
    
    # Create correlation heatmap
    if len(filtered_df[stats_columns]) > 1:
        corr_matrix = filtered_df[stats_columns].corr()
        heatmap_fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title='Correlation Heatmap'
        )
        heatmap_fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Features",
            coloraxis_colorbar=dict(title="Correlation")
        )
    else:
        heatmap_fig = go.Figure()
        heatmap_fig.update_layout(title="Not enough data for Correlation Heatmap.")
    
    # Perform PCA for dimensionality reduction
    if len(stats_columns) >= 2 and len(filtered_df) >= 2:
        try:
            pca = PCA(n_components=2)
            pca_results = pca.fit_transform(filtered_df[stats_columns])
            filtered_df['PCA1'] = pca_results[:, 0]
            filtered_df['PCA2'] = pca_results[:, 1]
            
            pca_fig = px.scatter(
                filtered_df,
                x='PCA1',
                y='PCA2',
                color='Cluster',
                title='PCA-based Clustering Visualization',
                hover_name='Company Name',
                size='Market Cap (USD Billion)',
                template='plotly_white',
                labels={'Cluster': 'Cluster'}
            )
            
            pca_fig.update_layout(
                xaxis_title='PCA Component 1',
                yaxis_title='PCA Component 2',
                legend_title_text='Cluster',
                legend=dict(
                    itemsizing='constant'
                )
            )
        except Exception as e:
            logger.error(f"PCA failed: {e}")
            pca_fig = go.Figure()
            pca_fig.update_layout(title="PCA could not be performed due to insufficient data.")
    else:
        pca_fig = go.Figure()
        pca_fig.update_layout(title="Not enough features or samples for PCA.")
    
    return scatter_fig, cluster_stats_html, heatmap_fig, pca_fig

# -------------------------------
# 7. Define Download Callback
# -------------------------------

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-button", "n_clicks"),
    [
        State('cluster-type', 'value'),
        State('num-clusters', 'value'),
        State('distance-metric', 'value'),
        State('linkage-method', 'value'),
        State('filter-industry', 'value'),
        State('filter-sector', 'value')
    ],
    prevent_initial_call=True
)
def download_clustered_data(n_clicks, cluster_type, num_clusters, distance_metric, linkage_method, selected_industries, selected_sectors):
    if n_clicks is None:
        return dash.no_update
    
    logger.info(f"Download requested for cluster type: {cluster_type}, k: {num_clusters}, distance_metric: {distance_metric}, linkage: {linkage_method}")
    logger.info(f"Selected industries: {selected_industries}")
    logger.info(f"Selected sectors: {selected_sectors}")
    
    # Apply filters
    filtered_df = df.copy()
    if selected_industries:
        filtered_df = filtered_df[filtered_df['Industry'].isin(selected_industries)]
    if selected_sectors:
        filtered_df = filtered_df[filtered_df['Sector'].isin(selected_sectors)]
    
    logger.info(f"Number of records after filtering: {len(filtered_df)}")
    
    if filtered_df.empty:
        logger.warning("No data to download after applying filters.")
        return dash.no_update
    
    # Define relevant columns for clustering based on cluster_type
    clustering_features = []
    if cluster_type == 'performance':
        clustering_features = performance_metrics
    elif cluster_type == 'peer':
        clustering_features = peer_dummies.columns.tolist()
    elif cluster_type == 'compensation':
        clustering_features = [
            'Salary (USD)', 'Bonus (USD)', 'Stock Awards (USD)',
            'Non-Equity Incentive Plan Compensation (USD)', 
            'All Other Compensation (USD)', 'Total CEO Compensation (Check)'
        ]
    elif cluster_type == 'individual':
        clustering_features = [
            'Salary (USD)', 'Bonus (USD)', 'Stock Awards (USD)',
            'Non-Equity Incentive Plan Compensation (USD)', 
            'All Other Compensation (USD)'
        ]
    else:
        clustering_features = performance_metrics  # Default
    
    # Ensure there are enough features and samples for clustering
    if len(clustering_features) < 1:
        logger.warning("No clustering features available for download.")
        return dash.no_update
    
    # Prepare data for clustering
    clustering_data = filtered_df[clustering_features]
    
    # Handle missing values by filling with median
    clustering_data = clustering_data.fillna(clustering_data.median())
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    # Perform Agglomerative Clustering
    try:
        clustering_model = AgglomerativeClustering(
            n_clusters=num_clusters,
            metric=distance_metric,
            linkage=linkage_method
        )
        cluster_labels = clustering_model.fit_predict(scaled_data)
        logger.info("Clustering completed successfully for download.")
    except ValueError as ve:
        logger.error(f"ValueError during clustering for download: {ve}")
        return dash.no_update
    except Exception as e:
        logger.error(f"Unexpected error during clustering for download: {e}")
        return dash.no_update
    
    # Assign cluster labels to the filtered DataFrame
    filtered_df['Cluster'] = cluster_labels  # Direct assignment without .values
    
    # Generate cluster names
    # Since AgglomerativeClustering doesn't provide centroids, compute the mean of features for each cluster
    cluster_centroids_sim = filtered_df.groupby('Cluster')[clustering_features].mean().reset_index()
    cluster_centroids_sim['Cluster'] = cluster_centroids_sim['Cluster'].astype(int)
    
    # Create a temporary dictionary to store centroids for naming
    cluster_centroids_download = {}
    cluster_centroids_download[cluster_type] = {}
    cluster_centroids_download[cluster_type][num_clusters] = cluster_centroids_sim.set_index('Cluster')
    
    cluster_names = generate_cluster_names(cluster_type, num_clusters)
    
    filtered_df['Cluster Name'] = filtered_df['Cluster'].apply(lambda x: cluster_names[x] if x < len(cluster_names) else f'Cluster {x + 1}')
    
    # Prepare CSV
    return dcc.send_data_frame(filtered_df.to_csv, "clustered_data.csv")

# -------------------------------
# 8. Run the Dash App
# -------------------------------

if __name__ == '__main__':
    app.run_server(port=8000,debug=True)
