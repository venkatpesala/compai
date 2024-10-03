import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import logging

# -------------------------------
# 1. Configure Logging
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------
# 2. Load and Preprocess Data
# -------------------------------

# Load the dataset from the Excel file
file_path = 'CompAI Peer Group Fundamentals 20240928 MSFT v.2.xlsx'
try:
    df = pd.read_excel(file_path)
    logger.info(f"Successfully loaded data from {file_path}")
except FileNotFoundError:
    logger.error(f"File {file_path} not found. Please check the path.")
    raise

# Handle missing data by replacing them with 0
df.fillna(0, inplace=True)

# Extract relevant columns for compensation signature
compensation_columns = [
    'Salary (USD)', 
    'Bonus (USD)', 
    'Stock Awards (USD)',
    'Non-Equity Incentive Plan Compensation (USD)', 
    'All Other Compensation (USD)', 
    'CEO Pay Ratio'
]

# Performance metrics
performance_metrics = ['Total Revenue Growth (%)']  # Updated to match the actual column name

# Ensure that the performance metrics exist in the dataset
for metric in performance_metrics:
    if metric not in df.columns:
        logger.error(f"'{metric}' column is missing from the dataset. It is required for model training.")
        raise ValueError(f"'{metric}' column is missing from the dataset. Please provide it.")

# Convert compensation columns to numeric, coercing errors to NaN
for col in compensation_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Handle missing values using KNN Imputer if there are missing values
if df[compensation_columns].isnull().values.any():
    logger.info("Missing values detected in compensation columns. Applying KNNImputer.")
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(df[compensation_columns])
    df[compensation_columns] = X_imputed
    logger.info("Missing values imputed using KNNImputer.")
else:
    logger.info("No missing values detected in compensation columns.")

# Extract peer group tickers and convert to dummy variables
if 'Peer Group Company Tickers' in df.columns:
    df['Peer Group Company Tickers'] = df['Peer Group Company Tickers'].fillna('')
    peer_dummies = df['Peer Group Company Tickers'].str.strip().str.get_dummies(sep=',').astype(int)
    df = pd.concat([df, peer_dummies], axis=1)
    logger.info("Peer group tickers converted to dummy variables.")
else:
    logger.warning("'Peer Group Company Tickers' column not found in data.")

# -------------------------------
# 3. Create Compensation Signatures
# -------------------------------

def create_compensation_signature(df, style='Proportion-based'):
    """
    Generates a compensation signature for each company based on the selected style.
    """
    if style == 'Proportion-based':
        scaler = MinMaxScaler()
        scaled_compensation = scaler.fit_transform(df[compensation_columns])
        compensation_signatures = pd.DataFrame(scaled_compensation, columns=compensation_columns)
        df['Compensation Signature'] = compensation_signatures.apply(lambda row: row.values, axis=1)
        logger.info("Compensation signatures created using Proportion-based style.")
    elif style == 'Performance-based':
        # Adjust Bonus and Stock Awards based on Total Revenue Growth
        df['Bonus Adjusted'] = df['Bonus (USD)'] * (1 + df['Total Revenue Growth (%)'] / 100)
        df['Stock Awards Adjusted'] = df['Stock Awards (USD)'] * (1 + df['Total Revenue Growth (%)'] / 100)
        adjusted_compensation = df[['Salary (USD)', 'Bonus Adjusted', 'Stock Awards Adjusted',
                                    'Non-Equity Incentive Plan Compensation (USD)', 
                                    'All Other Compensation (USD)']].copy()
        adjusted_compensation.rename(columns={'Bonus Adjusted': 'Bonus (USD)', 
                                             'Stock Awards Adjusted': 'Stock Awards (USD)'}, inplace=True)
        scaler = MinMaxScaler()
        scaled_compensation = scaler.fit_transform(adjusted_compensation)
        compensation_signatures = pd.DataFrame(scaled_compensation, columns=compensation_columns)
        df['Compensation Signature'] = compensation_signatures.apply(lambda row: row.values, axis=1)
        logger.info("Compensation signatures created using Performance-based style.")
    else:
        logger.error(f"Unknown compensation signature style: {style}")
        raise ValueError(f"Unknown compensation signature style: {style}")
    return df

# Initialize with Proportion-based style
df = create_compensation_signature(df, style='Proportion-based')

# -------------------------------
# 4. Train Prediction Models
# -------------------------------

# Prepare data for prediction
# Features: Compensation components
# Target: Total Revenue Growth (%)

# Filter companies with Total Revenue Growth > 0
df_model = df[df['Total Revenue Growth (%)'] > 0].copy()

if df_model.empty:
    logger.warning("No companies with Total Revenue Growth > 0 found. Using all available data for model training.")
    df_model = df.copy()

X = df_model[compensation_columns]
y = df_model['Total Revenue Growth (%)']

# Ensure no missing values after imputation
if X.isnull().values.any():
    logger.info("Missing values detected in features after initial imputation. Applying KNNImputer.")
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=compensation_columns)
    logger.info("Missing values imputed for model training.")
else:
    logger.info("No missing values in features for model training.")

# Feature Scaling (StandardScaler for better model performance)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train regression models
# Random Forest Regressor
regressor_rf = RandomForestRegressor(n_estimators=100, random_state=42)
regressor_rf.fit(X_train, y_train)
logger.info("RandomForestRegressor trained.")

# Linear Regression (for comparison)
regressor_lr = LinearRegression()
regressor_lr.fit(X_train, y_train)
logger.info("LinearRegression model trained.")

# Evaluate the models
y_pred_rf = regressor_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f"RandomForestRegressor Mean Squared Error on Test Set: {mse_rf:.2f}")
logger.info(f"RandomForestRegressor Mean Squared Error on Test Set: {mse_rf:.2f}")

y_pred_lr = regressor_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"LinearRegression Mean Squared Error on Test Set: {mse_lr:.2f}")
logger.info(f"LinearRegression Mean Squared Error on Test Set: {mse_lr:.2f}")

# -------------------------------
# 5. Initialize Dash App
# -------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deploying to platforms like Heroku

# -------------------------------
# 6. Define Helper Functions
# -------------------------------

def apply_compensation_style(base_firm, target_firm, style, df):
    """
    Applies the selected compensation style from target firm to base firm.
    """
    base_data = df[df['Company Name'] == base_firm][compensation_columns].values
    target_data = df[df['Company Name'] == target_firm][compensation_columns].values

    if len(base_data) == 0:
        logger.error(f"Base firm '{base_firm}' not found in the dataset.")
        raise ValueError(f"Base firm '{base_firm}' not found in the dataset.")
    if len(target_data) == 0:
        logger.error(f"Target firm '{target_firm}' not found in the dataset.")
        raise ValueError(f"Target firm '{target_firm}' not found in the dataset.")

    base_compensation = base_data[0]
    target_compensation = target_data[0]

    if style == 'Proportion-based':
        base_signature = base_compensation / base_compensation.sum()
        target_signature = target_compensation / target_compensation.sum()
        total_base_compensation = base_compensation.sum()
        new_compensation = target_signature * total_base_compensation
        logger.info(f"Applied Proportion-based style from '{target_firm}' to '{base_firm}'.")
    elif style == 'Performance-based':
        revenue_growth = df[df['Company Name'] == base_firm]['Total Revenue Growth (%)'].values[0]
        bonus_multiplier = 1 + revenue_growth / 100
        stock_awards_multiplier = 1 + revenue_growth / 100
        new_compensation = base_compensation.copy()
        bonus_idx = compensation_columns.index('Bonus (USD)')
        stock_awards_idx = compensation_columns.index('Stock Awards (USD)')
        new_compensation[bonus_idx] *= bonus_multiplier
        new_compensation[stock_awards_idx] *= stock_awards_multiplier
        # Normalize to maintain total compensation
        new_compensation /= new_compensation.sum() / base_compensation.sum()
        logger.info(f"Applied Performance-based style from '{target_firm}' to '{base_firm}'.")
    else:
        logger.error(f"Unknown compensation signature style: {style}")
        raise ValueError(f"Unknown compensation signature style: {style}")

    # Ensure no negative values
    new_compensation = np.clip(new_compensation, 0, None)

    # Create a DataFrame for adjusted compensation
    adjusted_df = pd.DataFrame([new_compensation], columns=compensation_columns)
    adjusted_df['Company Name'] = f"{base_firm} (Adjusted)"

    return adjusted_df

def predict_performance(compensation, model):
    """
    Predicts the Total Revenue Growth (%) based on compensation components using the provided model.
    """
    # Scale the compensation using the same scaler used during training
    compensation_scaled = scaler.transform([compensation])
    prediction = model.predict(compensation_scaled)[0]
    logger.info(f"Predicted Total Revenue Growth: {prediction:.2f}%")
    return prediction

def suggest_compensation(desired_growth, current_compensation, model, penalty_weight=0.0):
    """
    Suggests compensation adjustments to achieve the desired Total Revenue Growth (%).
    Includes a penalty to minimize drastic changes.
    """
    def objective(x):
        x = np.clip(x, 0, None)
        # Scale the new compensation
        x_scaled = scaler.transform([x])
        pred = model.predict(x_scaled)[0]
        growth_diff = (pred - desired_growth) ** 2
        penalty = np.sum(np.abs(x - current_compensation)) * penalty_weight
        return growth_diff + penalty

    initial_guess = current_compensation.copy()
    bounds = [(0, None) for _ in initial_guess]

    result = minimize(objective, initial_guess, bounds=bounds)

    if result.success:
        suggested_compensation = result.x
        logger.info(f"Suggested Compensation to achieve {desired_growth}% growth.")
    else:
        suggested_compensation = current_compensation
        logger.warning("Optimization failed. Returning current compensation.")

    return suggested_compensation

# -------------------------------
# 7. Define Dash Layout
# -------------------------------

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Compensation Signature Style Transfer"), width=12)
    ], justify='center', className='my-4'),
    
    dbc.Row([
        dbc.Col([
            html.Label('Select Base Company:'),
            dcc.Dropdown(
                id='base-company',
                options=[{'label': company, 'value': company} for company in df['Company Name'].unique()],
                value='Microsoft',  # Replace with an actual company from your dataset
                clearable=False
            ),
        ], md=3),
        
        dbc.Col([
            html.Label('Select Target Company:'),
            dcc.Dropdown(
                id='target-company',
                options=[{'label': company, 'value': company} for company in df['Company Name'].unique()],
                value='Tesla',  # Replace with an actual company from your dataset
                clearable=False
            ),
        ], md=3),
        
        dbc.Col([
            html.Label('Select Compensation Signature Style:'),
            dcc.Dropdown(
                id='signature-style',
                options=[
                    {'label': 'Proportion-based', 'value': 'Proportion-based'},
                    {'label': 'Performance-based', 'value': 'Performance-based'},
                    # Add more styles here if needed
                ],
                value='Proportion-based',
                clearable=False
            ),
        ], md=3),
        
        dbc.Col([
            html.Label(''),  # Empty label for alignment
            html.Button('Apply Compensation Style', id='apply-button', n_clicks=0, className="btn btn-primary", style={'width': '100%'}),
        ], md=3),
    ], className='mb-4'),
    
    dbc.Row([
        dbc.Col([
            html.H2("Original Compensation Signature (Base Company)"),
            dash_table.DataTable(
                id='base-signature-table',
                columns=[{"name": i, "id": i} for i in compensation_columns + ['Company Name']],
                data=[],
                style_cell={'textAlign': 'left'},
                style_table={'overflowX': 'auto'},
                page_size=10,
            ),
        ], md=6),
        
        dbc.Col([
            html.H2("Adjusted Compensation Signature (After Applying Target Style)"),
            dash_table.DataTable(
                id='adjusted-signature-table',
                columns=[{"name": i, "id": i} for i in compensation_columns + ['Company Name']],
                data=[],
                style_cell={'textAlign': 'left'},
                style_table={'overflowX': 'auto'},
                page_size=10,
            ),
        ], md=6),
    ], className='mb-4'),
    
    dbc.Row([
        dbc.Col([
            html.H2("Predict Company Performance Based on Compensation"),
            html.Label('Select Prediction Model:'),
            dcc.Dropdown(
                id='prediction-model',
                options=[
                    {'label': 'Random Forest Regressor', 'value': 'RandomForest'},
                    {'label': 'Linear Regression', 'value': 'LinearRegression'},
                ],
                value='RandomForest',
                clearable=False
            ),
            html.Br(),
            html.Button('Predict Performance', id='predict-button', n_clicks=0, className="btn btn-success"),
            html.Br(), html.Br(),
            html.Div(id='prediction-output'),
        ], md=6),
        
        dbc.Col([
            html.H2("Suggest Compensation Style for Desired Performance"),
            html.Label('Desired Revenue Growth (%):'),
            dcc.Input(id='desired-growth', type='number', value=10, step=0.1, min=0),
            html.Br(), html.Br(),
            html.Button('Suggest Compensation', id='suggest-button', n_clicks=0, className="btn btn-warning"),
            html.Br(), html.Br(),
            dash_table.DataTable(
                id='suggested-compensation-table',
                columns=[{"name": i, "id": i} for i in compensation_columns + ['Company Name']],
                data=[],
                style_cell={'textAlign': 'left'},
                style_table={'overflowX': 'auto'},
                page_size=10,
            ),
        ], md=6),
    ], className='mb-4'),
    
    # Download Section
    dbc.Row([
        dbc.Col([
            html.H2("Download Suggested Compensation"),
            html.Button("Download CSV", id="download-button", className="btn btn-info"),
            dcc.Download(id="download-dataframe-csv"),
        ], md=6),
    ], className='mb-4'),
    
    # Interactive Visualization
    dbc.Row([
        dbc.Col([
            html.H2("Compensation vs. Revenue Growth"),
            dcc.Graph(id='compensation-vs-growth'),
        ], md=12),
    ], className='mb-4'),
    
], fluid=True)

# -------------------------------
# 8. Define Callbacks
# -------------------------------

# Callback to apply compensation style and update tables
@app.callback(
    [Output('base-signature-table', 'data'),
     Output('adjusted-signature-table', 'data')],
    [Input('apply-button', 'n_clicks')],
    [State('base-company', 'value'),
     State('target-company', 'value'),
     State('signature-style', 'value')]
)
def update_compensation_tables(n_clicks, base_company, target_company, style):
    if n_clicks > 0:
        try:
            # Apply compensation style
            adjusted_compensation_df = apply_compensation_style(base_company, target_company, style, df)
            adjusted_compensation_dict = adjusted_compensation_df.to_dict('records')
            
            # Get base company's original compensation
            base_signature_df = df[df['Company Name'] == base_company][compensation_columns + ['Company Name']]
            base_signature_dict = base_signature_df.to_dict('records')
            
            logger.info(f"Updated compensation tables for '{base_company}' with style '{style}'.")
            return base_signature_dict, adjusted_compensation_dict
        except ValueError as ve:
            logger.error(f"ValueError: {ve}")
            return [], [{"Company Name": "Error", **{col: str(ve) for col in compensation_columns}}]
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return [], [{"Company Name": "Error", **{col: str(e) for col in compensation_columns}}]
    
    # Return empty data initially
    return [], []

# Callback to predict performance based on adjusted compensation
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('base-company', 'value'),
     State('target-company', 'value'),
     State('signature-style', 'value'),
     State('prediction-model', 'value')]
)
def predict_performance_callback(n_clicks, base_company, target_company, style, model_type):
    if n_clicks > 0:
        try:
            # Apply compensation style to get adjusted compensation
            adjusted_compensation_df = apply_compensation_style(base_company, target_company, style, df)
            adjusted_compensation = adjusted_compensation_df[compensation_columns].values[0]
            
            # Select model
            if model_type == 'RandomForest':
                model = regressor_rf
            elif model_type == 'LinearRegression':
                model = regressor_lr
            else:
                logger.warning(f"Unknown prediction model selected: {model_type}. Defaulting to RandomForest.")
                model = regressor_rf
            
            # Predict performance
            predicted_growth = predict_performance(adjusted_compensation, model)
            
            return html.Div([
                html.H4(f"Predicted Total Revenue Growth (%): {predicted_growth:.2f}")
            ])
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return html.Div([
                html.H4(f"Prediction failed: {str(e)}")
            ])
    
    return ""

# Callback to suggest compensation based on desired performance
@app.callback(
    Output('suggested-compensation-table', 'data'),
    [Input('suggest-button', 'n_clicks')],
    [State('desired-growth', 'value'),
     State('base-company', 'value')]
)
def suggest_compensation_callback(n_clicks, desired_growth, base_company):
    if n_clicks > 0:
        try:
            # Get current compensation of the base company
            current_compensation = df[df['Company Name'] == base_company][compensation_columns].values
            if len(current_compensation) == 0:
                raise ValueError(f"Base company '{base_company}' not found in the dataset.")
            current_compensation = current_compensation[0]
            
            # Suggest compensation adjustments
            suggested_compensation = suggest_compensation(desired_growth, current_compensation, regressor_rf)
            
            # Create a DataFrame for display
            suggested_comp_df = pd.DataFrame([suggested_compensation], columns=compensation_columns)
            suggested_comp_df['Company Name'] = f"{base_company} (Suggested)"
            
            logger.info(f"Suggested compensation adjustments for desired growth of {desired_growth}%.")
            return suggested_comp_df.to_dict('records')
        except Exception as e:
            logger.error(f"Suggestion error: {e}")
            return [{"Company Name": "Error", **{col: str(e) for col in compensation_columns}}]
    
    # Return empty data initially
    return []

# Callback to download suggested compensation as CSV
@app.callback(
    Output("download-dataframe-csv", "data"),
    [Input("download-button", "n_clicks")],
    [State('suggested-compensation-table', 'data')],
    prevent_initial_call=True,
)
def download_csv(n_clicks, data):
    if n_clicks and data:
        df_download = pd.DataFrame(data)
        logger.info("Downloading suggested compensation as CSV.")
        return dcc.send_data_frame(df_download.to_csv, "suggested_compensation.csv")
    return dash.no_update

# Callback for interactive visualization
@app.callback(
    Output('compensation-vs-growth', 'figure'),
    [Input('base-company', 'value')]
)
def update_graph(base_company):
    data = df[df['Company Name'] == base_company]
    if data.empty:
        logger.warning(f"No data available for company '{base_company}' to generate graph.")
        return {}
    fig = px.scatter(
        data, 
        x='Bonus (USD)', 
        y='Total Revenue Growth (%)',
        size='Stock Awards (USD)', 
        hover_name='Company Name',
        title=f'Bonus vs. Total Revenue Growth for {base_company}'
    )
    logger.info(f"Generated Compensation vs. Total Revenue Growth graph for '{base_company}'.")
    return fig

# -------------------------------
# 9. Run the Dash App
# -------------------------------

if __name__ == '__main__':
    app.run_server(debug=True)
