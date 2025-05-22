import dash
from dash import html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import tempfile

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Parameters
num_records = 500000
countries = ["UK", "USA", "Germany", "France", "Norway", "India", "Canada", "Japan", "Australia", "Brazil", "South Africa"]
resources = ["/index.html", "/scheduledemo.php", "/prototype.php", "/event.php", "/images/events.jpg"]
response_codes = [200, 404, 500, 304]
request_types = ["GET", "POST"]
sales_channels = ["Online", "Physical"]
salespersons = ["Ben", "Abishola", "Constance", "Peter", "Emma"]
retail_stores = ["Retail A", "Retail B", "Retail C", "Retail D"]
customer_segments = ["Small Business", "Enterprise", "Individual", "Startup"]

# Salesperson ID map
salesperson_ids = {name: f"SP{str(i+1).zfill(3)}" for i, name in enumerate(salespersons)}

# Country revenue multipliers
country_revenue_multiplier = {
    "UK": 1.1, "USA": 1.3, "Germany": 1.2, "France": 1.0, "Norway": 0.9,
    "India": 0.8, "Canada": 1.05, "Japan": 1.15, "Australia": 1.1,
    "Brazil": 0.85, "South Africa": 0.75
}

# Salesperson effectiveness factor
salesperson_effectiveness = {
    "Ben": 1.2,
    "Abishola": 1.1,
    "Constance": 1.3,
    "Peter": 0.95,
    "Emma": 1.05
}

# Product and industry setup
all_products = set(p for plist in {
    "Finance": ["AI Assistant Pro", "Market Forecaster", "Insight Analyzer"],
    "Healthcare": ["Smart HR Helper", "AI Assistant Pro", "Experience Tracker"],
    "Retail": ["Insight Analyzer", "Experience Tracker", "Rapid Proto Builder"],
    "Technology": ["DevOps AI Companion", "Rapid Proto Builder", "AI Assistant Pro"],
    "Education": ["Smart HR Helper", "Experience Tracker"],
    "Manufacturing": ["Market Forecaster", "DevOps AI Companion"],
    "Telecommunications": ["Insight Analyzer", "AI Assistant Pro"],
    "Logistics": ["Market Forecaster", "DevOps AI Companion", "Insight Analyzer"]
}.values() for p in plist)
product_ids = {product: f"P{str(i+1).zfill(4)}" for i, product in enumerate(all_products)}
industries_products = {
    "Finance": ["AI Assistant Pro", "Market Forecaster", "Insight Analyzer"],
    "Healthcare": ["Smart HR Helper", "AI Assistant Pro", "Experience Tracker"],
    "Retail": ["Insight Analyzer", "Experience Tracker", "Rapid Proto Builder"],
    "Technology": ["DevOps AI Companion", "Rapid Proto Builder", "AI Assistant Pro"],
    "Education": ["Smart HR Helper", "Experience Tracker"],
    "Manufacturing": ["Market Forecaster", "DevOps AI Companion"],
    "Telecommunications": ["Insight Analyzer", "AI Assistant Pro"],
    "Logistics": ["Market Forecaster", "DevOps AI Companion", "Insight Analyzer"]
}
industry_list = list(industries_products.keys())

# Generate data
data = []
for _ in range(num_records):
    timestamp = fake.date_time_this_year()
    ip_address = fake.ipv4()
    country = random.choice(countries)
    request_type = random.choice(request_types)
    resource = random.choice(resources)
    response_code = random.choice(response_codes)
    jobs_placed = np.random.randint(0, 10)
    demo_requests = np.random.randint(0, 5)
    ai_assistant_requests = np.random.randint(0, 3)
    sales_channel = random.choice(sales_channels)
    salesperson = random.choice(salespersons)
    salesperson_id = salesperson_ids[salesperson]
    retail_store = random.choice(retail_stores)

    # Revenue based on country and salesperson effectiveness
    base_revenue = np.random.randint(50, 1000)
    multiplier = country_revenue_multiplier[country] * salesperson_effectiveness[salesperson]
    revenue = int(base_revenue * multiplier)

    industry = random.choice(industry_list)
    product = random.choice(industries_products[industry])
    product_id = product_ids[product]

    customer_name = fake.company()
    customer_id = f"CUST{fake.unique.random_int(1000, 999999)}"
    customer_segment = random.choice(customer_segments)

    # Estimate average price per unit between $20 and $100 depending on product
    avg_price_per_unit = random.uniform(20, 100)
    sales = max(1, int(revenue / avg_price_per_unit))  # At least 1 unit sold

    data.append([
        timestamp, ip_address, country, request_type, resource, response_code,
        jobs_placed, demo_requests, ai_assistant_requests, sales_channel,
        salesperson, salesperson_id, retail_store, revenue, sales,
        product, product_id, industry, customer_name, customer_id, customer_segment
    ])
# Column headers
columns = ["Timestamp", "IP Address", "Country", "Request Type", "Resource Requested", "Response Code",
           "Jobs Placed", "Demo Requests", "AI Assistant Requests", "Sales Channel",
           "Salesperson", "Salesperson ID", "Retail Store", "Revenue", "Sales", "Product", "Product ID", "Industry",
           "Customer Name", "Customer ID", "Customer Segment"]

# Create and save DataFrame
df = pd.DataFrame(data, columns=columns)
df.to_csv("web_logs_data1.csv", index=False)

print("Dataset created and saved as 'web_logs_data.csv'")

#Save processed dataset (optional)
df.to_csv("web_logs_data1.csv", index=False)
print("Data preparation completed. Cleaned dataset saved as web_logs_data1.")

# Load your dataset
df_full = pd.read_csv('web_logs_data1.csv')

# Parse dates
if "Timestamp" in df_full.columns:
    df_full["Timestamp"] = pd.to_datetime(df_full["Timestamp"], errors='coerce')
    df_full["Date"] = pd.to_datetime(df_full["Timestamp"], errors='coerce').dt.date
else:
    df_full["Date"] = pd.to_datetime(df_full["Date"], errors='coerce')

# Prepare clustering data function
def prepare_clustering_data(df):
    try:
        country_stats = df.groupby('Country').agg({
            'Revenue': 'sum',
            'Demo Requests': 'sum',
            'Salesperson': 'nunique'
        }).reset_index()
        country_stats.rename(columns={'Salesperson': 'Unique_Salespersons'}, inplace=True)

        rep_perf = df.groupby(['Country', 'Salesperson'])['Revenue'].sum().reset_index()
        avg_rev = rep_perf.groupby('Country')['Revenue'].mean().reset_index()
        avg_rev.rename(columns={'Revenue': 'Avg_Rev_per_Salesperson'}, inplace=True)

        merged_df = pd.merge(country_stats, avg_rev, on='Country')
        features = ['Revenue', 'Demo Requests', 'Unique_Salespersons', 'Avg_Rev_per_Salesperson']
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(merged_df[features])
        return merged_df, scaled_features
    except:
        return pd.DataFrame(), np.array([])

# Initialize Dash app with suppress_callback_exceptions=True
app = dash.Dash(__name__, 
                external_stylesheets=[
                    dbc.themes.BOOTSTRAP,
                    "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
                ],
                suppress_callback_exceptions=True)
app.title = "Sales Dashboard"
server = app.server

# Layout
app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dbc.Row([
        # Sidebar
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

# Callbacks
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

@app.callback(
    [Output('total-revenue', 'children'),
     Output('total-sales', 'children')],
    [Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_totals(start_date, end_date):
    df = df_full.copy()
    if "Timestamp" in df.columns:
        df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
    else:
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    total_rev = df['Revenue'].sum() if 'Revenue' in df.columns else 0
    total_sales = len(df)
    return [
        f"Total Revenue: ${total_rev:,.2f}",
        f"Total Sales: {total_sales}"
    ]

@app.callback(
    Output('tab-content', 'children'),
    [Input('url', 'pathname'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def render_tab_content(pathname, start_date, end_date):
    df = df_full.copy()
    if "Timestamp" in df.columns:
        df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
    else:
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

    # Overview Page
    if pathname == "/visualization-overview" or pathname == "/":
        # Example: Sales by Region
        if not df.empty and "Country" in df.columns:
            region_sales = df.groupby("Country")["Sales"].sum().reset_index()
            fig_sales_by_region = px.bar(region_sales, x='Country', y='Sales', title='Sales by Region')
        else:
            fig_sales_by_region = px.bar()

        # Example: Sales by Salesperson
        if not df.empty and "Salesperson" in df.columns:
            sp_sales = df.groupby("Salesperson")["Sales"].sum().reset_index()
            fig_salesperson = px.bar(sp_sales, x='Salesperson', y='Sales', title='Sales by Salesperson')
            fig_salesperson.add_trace(go.Scatter(
                x=sp_sales['Salesperson'], y=[1e6]*len(sp_sales),
                mode='lines', name='Target', line=dict(color='red', width=1)
            ))
        else:
            fig_salesperson = px.bar()

        # Total Sales Over Time
        if not df.empty and "Date" in df.columns:
            total_sales_time = df.groupby("Date")["Sales"].sum().reset_index()
            fig_total_sales = px.line(total_sales_time, x='Date', y='Sales', title='Total Sales Over Time')

            # Dummy series for illustration
            dates = pd.to_datetime(total_sales_time['Date'])
            np.random.seed(42)
            n_points = len(dates)
            target_bases = np.random.uniform(1e4, 4e4, n_points)
            target_fluct = target_bases * 0.05
            target_series = target_bases + np.random.normal(0, target_fluct)
            expenditure_bases = np.random.uniform(2e4, 3e4, n_points)
            expenditure_fluct = expenditure_bases * 0.05
            expenditure_series = expenditure_bases + np.random.normal(0, expenditure_fluct)

            fig_total_sales.add_trace(go.Scatter(x=dates, y=total_sales_time['Sales'], mode='lines', name='Actual'))
            fig_total_sales.add_trace(go.Scatter(x=dates, y=target_series, mode='lines', name='Target'))
            fig_total_sales.add_trace(go.Scatter(x=dates, y=expenditure_series, mode='lines', name='Expenditure'))

        else:
            fig_total_sales = px.line()

        # Top products
        if not df.empty and "Product" in df.columns:
            top_products = df.groupby("Product")["Sales"].sum().reset_index().sort_values(by="Sales", ascending=False)
            np.random.seed(42)
            target_series_products = np.random.uniform(0.5e6, 1e6, size=len(top_products))
            target_series_products_fluct = target_series_products * 0.05
            target_series_products += np.random.normal(0, target_series_products_fluct)

            fig_top_products = go.Figure()
            fig_top_products.add_trace(go.Bar(
                x=top_products["Product"], y=top_products["Sales"], name='Sales'
            ))
            fig_top_products.add_trace(go.Scatter(
                x=top_products["Product"], y=target_series_products,
                mode='lines+markers', name='Target',
                line=dict(color='red', width=2)
            ))
            fig_top_products.update_layout(
                title='Top Selling Products with Target',
                xaxis_title='Product', yaxis_title='Sales'
            )
        else:
            fig_top_products = go.Figure()

        return html.Div([
            html.H3("Sales Manager Overview"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_salesperson), width=4),
                dbc.Col(dcc.Graph(figure=fig_total_sales), width=4),
                dbc.Col(dcc.Graph(figure=fig_top_products), width=4),
            ], className="g-0")
        ])

    # Data page
    elif pathname == "/data":
        return html.Div([
            html.H4("Data Overview"),
            dash_table.DataTable(
                id='data-table',
                data=df.sample(6, random_state=42).to_dict('records'),
                columns=[{"name": c, "id": c} for c in df.columns],
                page_size=6,
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#1f2c56', 'color': 'white', 'fontWeight': 'bold'},
                style_cell={'padding': '10px', 'textAlign': 'left', 'backgroundColor': '#2c3e50', 'color': 'white'}
            )
        ])

    # Sales performance
    elif pathname == "/sales-performance":
        if not df.empty and "Salesperson" in df.columns:
            sp_sales = df.groupby("Salesperson")["Sales"].sum().reset_index()
            fig_sales = px.bar(sp_sales, x='Salesperson', y='Sales', title='Sales by Salesperson')
            fig_sales.add_trace(go.Scatter(
                x=sp_sales['Salesperson'], y=[1e6]*len(sp_sales),
                mode='lines', name='Target', line=dict(color='red', width=1)
            ))
        else:
            fig_sales = px.bar()

        if not df.empty and "Sales Channel" in df.columns:
            fig_channel = px.pie(df, names="Sales Channel", values="Sales")
        else:
            fig_channel = px.pie()

        if not df.empty and "Country" in df.columns:
            region_sales = df.groupby("Country")["Sales"].sum().reset_index()
            fig_region = px.bar(region_sales, x='Country', y='Sales')
        else:
            fig_region = px.bar()

        return dbc.Container([
            html.H4("Sales Representatives"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_sales), width=4),
                dbc.Col(dcc.Graph(figure=fig_channel), width=4),
                dbc.Col(dcc.Graph(figure=fig_region), width=4),
            ])
        ])

    # Team Performance
    elif pathname == "/team-performance":
        store_perf = px.bar(df.groupby("Retail Store")["Sales"].sum().reset_index(), x="Retail Store", y="Sales")
        revenue_user = px.histogram(df.groupby("Customer Name")["Revenue"].sum().reset_index(), x="Customer Name", y="Revenue")
        conv = go.Figure(go.Indicator(mode="gauge+number", value=56.56, title={'text':"Conversion Rate (%)"},
                                    gauge={'axis': {'range':[0,100]}, 'bar': {'color':'#17BECF'}}))
        return html.Div([
            html.H4("Performance Analyst"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=store_perf), width=4),
                dbc.Col(dcc.Graph(figure=revenue_user), width=4),
                dbc.Col(dcc.Graph(figure=conv), width=4),
            ])
        ])

    # Product insights
    elif pathname == "/product-insights":
        top_products = df.groupby("Product")["Sales"].sum().reset_index().sort_values(by="Sales", ascending=False)
        np.random.seed(42)
        target_series_products = np.random.uniform(0.5e6, 1e6, size=len(top_products))
        target_series_products_fluct = target_series_products * 0.05
        target_series_products += np.random.normal(0, target_series_products_fluct)

        dates = pd.to_datetime(df.groupby("Date")["Revenue"].sum().reset_index()["Date"])
        n_points = len(dates)
        target_bases = np.random.uniform(1e4, 4e4, n_points)
        target_fluct = target_bases * 0.05
        target_series_time = target_bases + np.random.normal(0, target_fluct)
        expenditure_bases = np.random.uniform(1e4, 4e4, n_points)
        expenditure_fluct = expenditure_bases * 0.05
        expenditure_series = expenditure_bases + np.random.normal(0, expenditure_fluct)

        total_sales_time = pd.DataFrame({'Date': dates, 'Sales': df.groupby("Date")["Sales"].sum().values})
        fig_time = px.line(total_sales_time, x='Date', y='Sales', title="Total Sales Over Time")
        fig_time.add_trace(go.Scatter(x=dates, y=total_sales_time['Sales'], mode='lines', name='Actual'))
        fig_time.add_trace(go.Scatter(x=dates, y=target_series_time, mode='lines', name='Target'))
        fig_time.add_trace(go.Scatter(x=dates, y=expenditure_series, mode='lines', name='Expenditure'))

        fig_products = px.bar(top_products, x="Product", y="Sales")
        # Add target line
        fig_products.add_trace(go.Scatter(
            x=top_products["Product"], y=target_series_products,
            mode='lines', name='Targets', line=dict(color='red', width=2)
        ))

        return html.Div([
            html.H4("Product Insights"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_products), width=6),
                dbc.Col(dcc.Graph(figure=fig_time), width=6),
            ])
        ])

    # Cluster analysis
    elif pathname == "/cluster-analysis":
        try:
            merge_df, scaled_features = prepare_clustering_data(df)
            n_clusters = 3
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(scaled_features)
                scores = {
                    "Silhouette": silhouette_score(scaled_features, labels),
                    "Davies-Bouldin": davies_bouldin_score(scaled_features, labels),
                    "Calinski-Harabasz": calinski_harabasz_score(scaled_features, labels)
                }
                merge_df['Cluster'] = labels.astype(str)
                fig_scatter = px.scatter(
                    merge_df, x=scaled_features[:,0], y=scaled_features[:,1],
                    color='Cluster', title='Cluster Scatter Plot'
                )
            except:
                scores = {"Error": "Clustering failed"}
                fig_scatter = go.Figure()
                fig_scatter.update_layout(title="Clustering failed")
        except:
            scores = {"Error": "Data preparation failed"}
            fig_scatter = go.Figure()
            fig_scatter.update_layout(title="Data preparation failed")

        key_info = (
            "0 : Country-based sales clusters  "
            "1 : Demo request trend  "
            "2 : Salesperson performance"
        )
        fig_scatter.update_layout(
            annotations=[
                dict(
                    text=key_info,
                    x=0.5,
                    y=-0.2,
                    xref='paper',
                    yref='paper',
                    showarrow=False,
                    align='left',
                    font=dict(size=10),
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1,
                    opacity=0.8
                )
            ],
            margin=dict(b=100)
        )

        def generate_score_html(scores):
            if "Error" in scores:
                return html.P(scores["Error"], style={'color': 'red'})
            return html.Div([
                html.Span(f"Silhouette: {scores['Silhouette']:.2f}  "),
                html.Span(f"Davies-Bouldin: {scores['Davies-Bouldin']:.2f}  "),
                html.Span(f"Calinski-Harabasz: {scores['Calinski-Harabasz']:.2f}")
            ], style={'fontSize': '14px', 'color': '#333', 'whiteSpace': 'nowrap'})

        return html.Div([
            html.Div([
                dcc.Graph(figure=fig_scatter),
                generate_score_html(scores)
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
        ])

    # Default fallback
    return html.H3("Page not found or no data.")


# Search callback
@app.callback(
    Output('data-table', 'data'),
    [Input('search-btn', 'n_clicks')],
    [State('search-input', 'value'), State('search-column', 'value')],
    prevent_initial_call=True
)
def filter_table(n_clicks, search_value, search_column):
    if not search_value or search_value.strip() == "":
        return df_full.sample(6, random_state=42).to_dict('records')
    filtered = df_full[df_full[search_column].astype(str).str.lower().str.contains(search_value.lower())]
    return filtered.to_dict('records') if not filtered.empty else []

# Reset search input
@app.callback(
    Output('search-input', 'value'),
    [Input('url', 'pathname')],
    prevent_initial_call=True
)
def reset_search(_):
    return ""

# Download PDF callback
@app.callback(
    Output("download-pdf", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def generate_pdf(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    import io
    import os
    import tempfile
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader

    with tempfile.TemporaryDirectory() as tmpdirname:
        pdf_path = os.path.join(tmpdirname, "dashboard_charts.pdf")
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter

        # List your charts here, matching your dashboard code
        # Example: replace these with the actual figures you generate
        figures = [
            ("Sales by Region", px.bar(
                df_full.groupby("Country")["Sales"].sum().reset_index(),
                x="Country", y="Sales"
            )),
            ("Sales by Salesperson", px.bar(
                df_full.groupby("Salesperson")["Sales"].sum().reset_index(),
                x="Salesperson", y="Sales"
            )),
            ("Total Sales Over Time", px.line(
                df_full.groupby("Date")["Sales"].sum().reset_index(),
                x="Date", y="Sales"
            )),
            ("Top Products", px.bar(
                df_full.groupby("Product")["Sales"].sum().reset_index().sort_values(by="Sales", ascending=False)
            )),
            ("Revenue per User", px.histogram(
                df_full.groupby("Customer Name")["Sales"].sum().reset_index(),
                x="Customer Name", y="Sales"
            )),
            ("Demo Requests & Conversion", go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=56.56,
                    title={'text': "Demo Requests (%)"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#17BECF"}}
                )
            ))
        ]

        descriptions = [
            "Revenue by Country",
            "Sales by Salesperson",
            "Total Sales Over Time",
            "Top Selling Products",
            "Revenue per Customer",
            "Demo Request Conversion Rate"
        ]

        for i, (title, fig) in enumerate(figures):
            # Convert plotly figure to image (png)
            img_bytes = fig.to_image(format='png', width=800, height=600)
            image = ImageReader(io.BytesIO(img_bytes))
            # Draw title and description
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height - 50, title)
            c.setFont("Helvetica", 10)
            c.drawString(50, height - 70, descriptions[i])
            # Draw the image
            c.drawImage(image, 50, height - 650, width=500, height=300, preserveAspectRatio=True)
            c.showPage()

        c.save()

        # Read the PDF bytes
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        return dash.dcc.send_bytes(lambda: pdf_bytes, filename="Dashboard_Charts.pdf")



if __name__ == '__main__':
    # Don't run in debug mode
    app.run(host='0.0.0.0', port=8051, debug=False)

