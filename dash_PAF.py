import pandas as pd
import plotly.graph_objects as go
import plotly.colors
from dash import Dash, dcc, html, Input, Output, State
import os

CSV_FOLDER = './out'
DEFAULT_FILE = '2025_6.csv'

app = Dash(__name__)
app.title = "Price and Forecasts"

def list_csv_files():
    return [f for f in os.listdir(CSV_FOLDER) if f.endswith('.csv')]

def load_and_prepare_data(filename):
    path = os.path.join(CSV_FOLDER, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, decimal=',', parse_dates=['data'])
    df = df.sort_values('data')
    for col in df.columns:
        if col != 'data':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

app.layout = html.Div([
    html.H1("Price and Forecasts Viewer"),

    html.Div([
        html.Label("Выберите CSV файл:"),
        dcc.Dropdown(
            id='file-selector',
            options=[{"label": f, "value": f} for f in list_csv_files()],
            value=DEFAULT_FILE,
            clearable=False
        )
    ], style={'marginBottom': '10px'}),

    html.Button("Обновить график", id='update-button', n_clicks=0),

    dcc.Graph(id='price-graph', style={'height': '600px'}),

    # Интервал для автообновления
    dcc.Interval(
        id='interval-component',
        interval=1 * 60 * 1000,  # 1 минут
        n_intervals=0
    ),

    # Хранилище масштаба
    dcc.Store(id='stored-zoom', data={})
])

# Обновление графика по кнопке и автоинтервалу
@app.callback(
    Output('price-graph', 'figure'),
    Input('update-button', 'n_clicks'),
    Input('interval-component', 'n_intervals'),
    State('file-selector', 'value'),
    State('stored-zoom', 'data')
)
def update_graph(n_clicks, n_intervals, selected_file, zoom_data):
    df = load_and_prepare_data(selected_file)
    fig = go.Figure()

    if df.empty:
        fig.update_layout(title="Нет данных")
        return fig

    colors = plotly.colors.qualitative.Plotly
    num_colors = len(colors)

    for i, col in enumerate(df.columns):
        if col != 'data':
            fig.add_trace(go.Scatter(
                x=df['data'],
                y=df[col],
                mode='lines+markers',
                name=col,
                line=dict(
                    color="black" if col == "praice" else colors[i % num_colors],
                    width=2 if col == "praice" else 1,
                    dash='solid' if col == "praice" else 'dot'
                ),
                marker=dict(size=5)
            ))

    # fig.update_layout(
    #     title='Price and Forecasts',
    #     xaxis_title='Date',
    #     yaxis_title='Price',
    #     template='plotly_white',
    #     hovermode='x unified',
    #     yaxis=dict(autorange=True),
    #     legend=dict(
    #         orientation="h",
    #         yanchor="bottom",
    #         y=-0.3,
    #         xanchor="center",
    #         x=0.5
    #     )
    # )
    fig.update_layout(
        title='Price and Forecasts',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        hovermode='x unified',
        yaxis=dict(autorange=True),
        margin=dict(t=40, b=100),  # t — сверху, b — снизу (увеличен отступ снизу)
        legend=dict(
            orientation="h",
            yanchor="top",    # теперь верхняя граница легенды приравнивается к y
            y=-0.35,          # смещаем ниже графика
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        )
    )
    

    if zoom_data:
        if 'xaxis.range' in zoom_data:
            fig.update_xaxes(range=zoom_data['xaxis.range'], autorange=False)
        if 'yaxis.range' in zoom_data:
            fig.update_yaxes(range=zoom_data['yaxis.range'], autorange=False)

    return fig

# Сохраняем масштаб при изменении графика
@app.callback(
    Output('stored-zoom', 'data'),
    Input('price-graph', 'relayoutData'),
    State('stored-zoom', 'data'),
    prevent_initial_call=True
)
def store_zoom(relayout_data, stored_data):
    stored_data = stored_data or {}
    new_data = {}

    if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
        new_data['xaxis.range'] = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
    else:
        new_data['xaxis.range'] = stored_data.get('xaxis.range')

    if 'yaxis.range[0]' in relayout_data and 'yaxis.range[1]' in relayout_data:
        new_data['yaxis.range'] = [relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]
    else:
        new_data['yaxis.range'] = stored_data.get('yaxis.range')

    return new_data

if __name__ == "__main__":
    app.run(debug=True)
