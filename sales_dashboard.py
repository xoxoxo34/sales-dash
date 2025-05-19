#FINAL CODE
import dash
from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64
import io
import tempfile
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# --- Begin clustering analysis (precompute at startup) ---
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Load dataset
df = pd.read_csv('web_logs_data1.csv')

# Aggregate core metrics per country
country_stats = df.groupby('Country').agg({
    'Revenue': 'sum',
    'Demo Requests': 'sum',
    'Salesperson': 'nunique'
}).reset_index()
country_stats.rename(columns={'Salesperson': 'Unique_Salespersons'}, inplace=True)

# Add average revenue per salesperson
rep_performance = df.groupby(['Country', 'Salesperson'])['Revenue'].sum().reset_index()
avg_rev_per_salesperson = rep_performance.groupby('Country')['Revenue'].mean().reset_index()
avg_rev_per_salesperson.rename(columns={'Revenue': 'Avg_Rev_per_Salesperson'}, inplace=True)

# Merge into one dataframe
merged_df = pd.merge(country_stats, avg_rev_per_salesperson, on='Country')

# Scale features
features = ['Revenue', 'Demo Requests', 'Unique_Salespersons', 'Avg_Rev_per_Salesperson']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(merged_df[features])

# Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
merged_df['Cluster'] = kmeans.fit_predict(scaled_features)

# Optional: Evaluate clustering
silhouette = silhouette_score(scaled_features, merged_df['Cluster'])
db_score = davies_bouldin_score(scaled_features, merged_df['Cluster'])
ch_score = calinski_harabasz_score(scaled_features, merged_df['Cluster'])
print(f"Silhouette Score: {silhouette:.2f}")
print(f"Davies-Bouldin Index: {db_score:.2f}")
print(f"Calinski-Harabasz Index: {ch_score:.2f}")

# Create Plotly scatter plot for clusters
cluster_fig = px.scatter(
    merged_df,
    x='Revenue',
    y='Demo Requests',
    color='Cluster',
    hover_name='Country',
    title='Clustering of Countries Based on Sales & Demo Insights',
    labels={'Revenue': 'Total Revenue', 'Demo Requests': 'Total Demo Requests'}
)

# --- End clustering analysis ---

# Load your dataset
df_full = pd.read_csv('web_logs_data1.csv')

# Parse Timestamp if exists
if "Timestamp" in df_full.columns:
    df_full["Timestamp"] = pd.to_datetime(df_full["Timestamp"], errors='coerce')
    df_full["Date"] = df_full["Timestamp"].dt.date

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.BOOTSTRAP,
    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
])
app.title = "Sales Dashboard"

# --- Define the layout ---
app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dbc.Row([
        # Sidebar with nav
        dbc.Col([
            html.Div([
                html.H5("Sales Dashboard", className="text-white mb-4"),
                dbc.Nav([
                    dbc.NavLink("Sales Manager", href="/visualization-overview", id="visualization-overview-tab", className="nav-link"),
                    dbc.NavLink("Data Overview", href="/data", id="data-overview-tab", className="nav-link"),
                    dbc.NavLink("Sales Representatives", href="/sales-performance", id="sales-performance-tab", className="nav-link"),
                    dbc.NavLink("Performance Analyst", href="/team-performance", id="team-performance-tab", className="nav-link"),
                    dbc.NavLink("Product Analyst", href="/product-insights", id="product-insights-tab", className="nav-link"),
                    dbc.NavLink("Cluster Analysis", href="/cluster-analysis", id="cluster-analysis-tab", className="nav-link"),
                ], vertical=True, pills=False, id="tabs-nav"),
                html.Hr(),
                html.Label("Search Data", className="text-light mt-2"),
                dbc.InputGroup([
                    dbc.Select(
                        id='search-column',
                        options=[{"label": col, "value": col} for col in df_full.columns],
                        value=df_full.columns[0],
                        style={"maxWidth": "150px"}
                    ),
                    dbc.Input(id='search-input', placeholder='Search...', debounce=True),
                    dbc.Button(html.I(className="bi bi-search"), id='search-btn', color='secondary')
                ], className="mb-3", style={"marginTop": "10px"}),
                html.Div(id="download-section", children=[
                    dbc.Button("Download PDF", id="download-btn", color="info", className="mt-2"),
                    dcc.Download(id="download-pdf")
                ])
            ], style={
                'background-color': '#1f2c56',
                'padding': '20px',
                'height': '100vh',
                'border-radius': '10px',
                'box-shadow': '2px 2px 10px rgba(0,0,0,0.4)'
            })
        ], width=3),
        # Main content
        dbc.Col([
            # Date range filter and stats
            dbc.Row([
                dbc.Col([
                    html.Label("Select Date Range"),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=df_full['Timestamp'].min() if "Timestamp" in df_full.columns else df_full['Date'].min(),
                        end_date=df_full['Timestamp'].max() if "Timestamp" in df_full.columns else df_full['Date'].max(),
                        display_format='YYYY-MM-DD'
                    )
                ], width=4),
                dbc.Col([
                    html.Div(id='total-revenue', style={'fontSize': '20px', 'marginTop': '10px'}),
                    html.Div(id='total-sales', style={'fontSize': '20px', 'marginTop': '10px'})
                ], width=8)
            ], style={'marginBottom': '20px'}),
            # Tab content
            html.Div(id='tab-content', style={
                'padding': '20px',
                'background-color': '#f4f6f9',
                'border': '1px solid #ced4da',
                'border-radius': '10px'
            })
        ], width=9)
    ])
], fluid=True)


# --- Highlight active tab ---
@app.callback(
    [Output('visualization-overview-tab', 'className'),
     Output('data-overview-tab', 'className'),
     Output('sales-performance-tab', 'className'),
     Output('team-performance-tab', 'className'),
     Output('product-insights-tab', 'className'),
     Output('cluster-analysis-tab', 'className')],
    [Input('url', 'pathname')]
)
def update_active_links(pathname):
    return [
        "nav-link" + (" active" if (pathname == "/visualization-overview" or pathname == "/") else ""),
        "nav-link" + (" active" if pathname == "/data" else ""),
        "nav-link" + (" active" if pathname == "/sales-performance" else ""),
        "nav-link" + (" active" if pathname == "/team-performance" else ""),
        "nav-link" + (" active" if pathname == "/product-insights" else ""),
        "nav-link" + (" active" if pathname == "/cluster-analysis" else "")
    ]


# --- Update total revenue and sales based on date filter ---
@app.callback(
    [Output('total-revenue', 'children'),
     Output('total-sales', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_totals(start_date, end_date):
    filtered_df = df_full.copy()
    if "Timestamp" in df_full.columns:
        filtered_df = filtered_df[
            (filtered_df['Timestamp'] >= start_date) & (filtered_df['Timestamp'] <= end_date)
        ]
    else:
        filtered_df = filtered_df[
            (filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)
        ]
    total_rev = filtered_df['Revenue'].sum() if 'Revenue' in filtered_df.columns else 0
    total_sales_count = len(filtered_df)
    return [
        f"Total Revenue: ${total_rev:,.2f}",
        f"Total Sales: {total_sales_count}"
    ]


# --- Render page content based on URL ---
@app.callback(
    Output('tab-content', 'children'),
    [Input('url', 'pathname'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def render_tab_content(pathname, start_date, end_date):
    # Filter data
    filtered_df = df_full.copy()
    if "Timestamp" in df_full.columns:
        filtered_df = filtered_df[
            (filtered_df['Timestamp'] >= start_date) & (filtered_df['Timestamp'] <= end_date)
        ]
    else:
        filtered_df = filtered_df[
            (filtered_df['Date'] >= start_date) & (filtered_df['Date'] <= end_date)
        ]

    if pathname == "/visualization-overview" or pathname == "/":
        # Dashboard overview with charts
        # 1. Sales by Salesperson
        sp_sales = filtered_df.groupby("Salesperson")["Revenue"].sum().reset_index()
        fig_salesperson = px.bar(sp_sales, x="Salesperson", y="Revenue", title="Sales by Salesperson")
        fig_salesperson.add_trace(go.Scatter(
            x=sp_sales["Salesperson"],
            y=[50_000_000]*len(sp_sales),
            mode='lines',
            name='Target',
            line=dict(color='red', width=1)
        ))

        # 2. Total Sales Over Time
        total_sales_time = filtered_df.groupby("Date")["Revenue"].sum().reset_index()
        fig_total_sales = px.line(total_sales_time, x="Date", y="Revenue", title="Total Sales Over Time")
        # Generate dummy actual, target, expenditure series
        dates = pd.to_datetime(total_sales_time["Date"])
        np.random.seed(42)
        n_points = len(dates)
        target_bases = np.random.uniform(1e6, 2.5e6, n_points)
        target_fluct = target_bases * 0.05
        target_series = target_bases + np.random.normal(0, target_fluct)
        expenditure_bases = np.random.uniform(5e5, 2e6, n_points)
        expenditure_fluct = expenditure_bases * 0.05
        expenditure_series = expenditure_bases + np.random.normal(0, expenditure_fluct)
        fig_total_sales.add_trace(go.Scatter(x=dates, y=total_sales_time["Revenue"], mode='lines', name='Actual'))
        fig_total_sales.add_trace(go.Scatter(x=dates, y=target_series, mode='lines', name='Target'))
        fig_total_sales.add_trace(go.Scatter(x=dates, y=expenditure_series, mode='lines', name='Expenditure'))

        # 3. Top Selling Products
        top_products = filtered_df.groupby("Product")["Revenue"].sum().reset_index().sort_values(by="Revenue", ascending=False)
        np.random.seed(42)
        target_series_products = np.random.uniform(3e7, 5e7, size=len(top_products))
        target_series_products_fluct = target_series_products * 0.05
        target_series_products = target_series_products + np.random.normal(0, target_series_products_fluct)
        fig_top_products = px.bar(top_products, x="Product", y="Revenue", title="Top Selling Products")
        fig_top_products.add_trace(go.Scatter(
            x=top_products["Product"],
            y=target_series_products,
            mode='lines',
            name='Targets',
            line=dict(color='red', width=2)
        ))

        return html.Div([
            html.H3("Sales Manager Overview"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_salesperson), width=4),
                dbc.Col(dcc.Graph(figure=fig_total_sales), width=4),
                dbc.Col(dcc.Graph(figure=fig_top_products), width=4),
            ], className="g-0")
        ])

    elif pathname == "/data":
        return html.Div([
            html.H4("Data Overview", className="mb-3 text-primary"),
            html.P("Explore the dataset or see a sample of entries below."),
            dash_table.DataTable(
                id='data-table',
                data=df_full.sample(n=6, random_state=42).to_dict('records'),
                columns=[{"name": i, "id": i} for i in df_full.columns],
                page_size=6,
                style_table={'overflowX': 'auto', 'margin': '0 auto'},
                style_header={'backgroundColor': '#1f2c56', 'color': 'white', 'fontWeight': 'bold'},
                style_cell={'padding': '10px', 'textAlign': 'left', 'minWidth': '100px', 'backgroundColor': 'white', 'border': '1px solid #ccc'},
                style_data={'border': '1px solid #ccc', 'backgroundColor': '#2c3e50', 'color': 'white'},
                style_as_list_view=True
            )
        ])

    elif pathname == "/sales-performance":
        # Sales by Salesperson
        sp_sales = filtered_df.groupby("Salesperson")["Revenue"].sum().reset_index()
        fig_sales = px.bar(sp_sales, x="Salesperson", y="Revenue", title="Sales by Salesperson")
        fig_sales.add_trace(go.Scatter(
            x=sp_sales["Salesperson"],
            y=[50_000_000]*len(sp_sales),
            mode='lines',
            name='Target',
            line=dict(color='red', width=1)
        ))

        # Sales by Channel pie
        fig_channel = px.pie(filtered_df, names="Sales Channel", values="Revenue", title="Sales by Channel")
        return dbc.Container([
            html.H4("Sales Representatives", className="text-primary mb-4"),
            html.Div(f"Total Sales: ${filtered_df['Revenue'].sum():,.2f}", className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_sales), md=6),
                dbc.Col(dcc.Graph(figure=fig_channel), md=6),
            ])
        ])

    elif pathname == "/team-performance":
        total_sales = filtered_df["Revenue"].sum()
        store_perf = px.bar(filtered_df.groupby("Retail Store")["Revenue"].sum().reset_index(), x="Retail Store", y="Revenue", title="Retail Store Performance")
        revenue_user = px.histogram(filtered_df.groupby("Customer Name")["Revenue"].sum().reset_index(), x="Revenue", title="Revenue per User")
        # Dummy conversion indicator
        conversion = go.Figure(go.Indicator(
            mode="gauge+number",
            value=56.56,
            title={'text': "Conversion Rate (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#17BECF"}}
        ))
        return html.Div([
            html.H4("Performance Analyst", className="mb-3 text-primary"),
            html.Div(f"Total Sales: ${total_sales:,.2f}", className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=store_perf), width=4),
                dbc.Col(dcc.Graph(figure=revenue_user), width=4),
                dbc.Col(dcc.Graph(figure=conversion), width=4),
            ])
        ])

    elif pathname == "/product-insights":
        total_sales = filtered_df["Revenue"].sum()
        top_products = filtered_df.groupby("Product")["Revenue"].sum().reset_index().sort_values(by="Revenue", ascending=False)
        # Generate target data
        np.random.seed(42)
        target_series_products = np.random.uniform(3e7, 5e7, size=len(top_products))
        target_series_products_fluct = target_series_products * 0.05
        target_series_products = target_series_products + np.random.normal(0, target_series_products_fluct)

        # Over time data
        dates = pd.to_datetime(filtered_df.groupby("Date")["Revenue"].sum().reset_index()["Date"])
        n_points = len(dates)
        target_bases = np.random.uniform(1e6, 2.5e6, n_points)
        target_fluct = target_bases * 0.05
        target_series_time = target_bases + np.random.normal(0, target_fluct)
        expenditure_bases = np.random.uniform(5e5, 2e6, n_points)
        expenditure_fluct = expenditure_bases * 0.05
        expenditure_series = expenditure_bases + np.random.normal(0, expenditure_fluct)

        total_sales_time = pd.DataFrame({'Date': dates, 'Revenue': filtered_df.groupby("Date")["Revenue"].sum().values})
        fig_total_sales_time = px.line(total_sales_time, x='Date', y='Revenue', title="Total Sales Over Time")
        fig_total_sales_time.add_trace(go.Scatter(x=dates, y=total_sales_time['Revenue'], mode='lines', name='Actual'))
        fig_total_sales_time.add_trace(go.Scatter(x=dates, y=target_series_time, mode='lines', name='Target'))
        fig_total_sales_time.add_trace(go.Scatter(x=dates, y=expenditure_series, mode='lines', name='Expenditure'))

        # Top Products with target line
        fig_top_products = px.bar(top_products, x="Product", y="Revenue", title="Top Selling Products")
        fig_top_products.add_trace(go.Scatter(
            x=top_products["Product"],
            y=target_series_products,
            mode='lines',
            name='Targets',
            line=dict(color='red', width=2)
        ))

        return html.Div([
            html.H4("Product Analyst", className="mb-3 text-primary"),
            html.Div(f"Total Sales: ${total_sales:,.2f}", className="mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_top_products), width=6),
                dbc.Col(dcc.Graph(figure=fig_total_sales_time), width=6),
            ])
        ])

    elif pathname == "/cluster-analysis":
        # Show the cluster plot
        return html.Div([
            html.H3("Cluster Analysis of Countries"),
            dcc.Graph(figure=cluster_fig)
        ])

    else:
        return html.H3("Page not found")


# --- Search callback ---
@app.callback(
    Output('data-table', 'data'),
    [Input('search-btn', 'n_clicks')],
    [State('search-input', 'value'), State('search-column', 'value')],
    prevent_initial_call=True
)
def filter_table(n_clicks, search_value, search_column):
    if not search_value or search_value.strip() == "":
        return df_full.sample(n=6, random_state=42).to_dict('records')
    filtered = df_full[df_full[search_column].astype(str).str.lower().str.contains(search_value.lower())]
    return filtered.to_dict('records') if not filtered.empty else []

@app.callback(
    Output('search-input', 'value'),
    [Input('url', 'pathname')],
    prevent_initial_call=True
)
def reset_search(n):
    return ""


# --- Generate PDF ---
from dash.exceptions import PreventUpdate
@app.callback(
    Output("download-pdf", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def generate_pdf(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    figures = [
        ("Top Selling Products", px.bar(
            df_full.groupby("Product")["Revenue"].sum().reset_index(),
            x="Product", y="Revenue")),
        ("Total Sales Over Time", px.line(
            df_full.groupby("Date")["Revenue"].sum().reset_index(),
            x="Date", y="Revenue")),
        ("Sales by Salesperson", px.bar(
            df_full.groupby("Salesperson")["Revenue"].sum().reset_index(),
            x="Salesperson", y="Revenue")),
        ("Sales by Channel", px.pie(
            df_full,
            names="Sales Channel",
            values="Revenue")),
        ("Sales by Region", px.bar(
            df_full.groupby("Country")["Revenue"].sum().reset_index(),
            x="Country", y="Revenue")),
        ("Retail Store Performance", px.bar(
            df_full.groupby("Retail Store")["Revenue"].sum().reset_index(),
            x="Retail Store", y="Revenue")),
        ("Revenue per User", px.histogram(
            df_full.groupby("Customer Name")["Revenue"].sum().reset_index(),
            x="Revenue")),
        ("Conversion Rate", go.Figure(go.Indicator(
            mode="gauge+number",
            value=56.56,
            title={'text': "Conversion Rate (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#17BECF"}}
        )))
    ]
    descriptions = [
        "This bar chart shows revenue by product.",
        "This line graph depicts total revenue trends over time.",
        "This bar chart presents revenue contribution by each salesperson.",
        "This pie chart shows revenue distribution across sales channels.",
        "This bar chart highlights revenue by region (country).",
        "Retail store performance in terms of total revenue is visualized here.",
        "This histogram represents revenue per individual user.",
        "This gauge chart shows the percentage of demo requests compared to total entries."
    ]
    with tempfile.TemporaryDirectory() as tmpdirname:
        c = canvas.Canvas(os.path.join(tmpdirname, "report.pdf"), pagesize=letter)
        width, height = letter
        for idx, (title, fig) in enumerate(figures):
            img_bytes = fig.to_image(format="png")
            image = ImageReader(io.BytesIO(img_bytes))
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, height - 50, title)
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 65, descriptions[idx])
            c.drawImage(image, 50, height - 420, width=500, height=300, preserveAspectRatio=True)
            c.showPage()
        c.save()
        pdf_path = os.path.join(tmpdirname, "report.pdf")
        with open(pdf_path, "rb") as f:
            encoded_pdf = base64.b64encode(f.read()).decode()
        return dcc.send_bytes(base64.b64decode(encoded_pdf), filename="Sales_Report.pdf")


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
