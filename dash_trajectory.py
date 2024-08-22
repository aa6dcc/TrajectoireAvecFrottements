import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from RungeKutta import secondOrderRungeKutta

app = dash.Dash(__name__, title='2D Trajectory with Drag')

inputWidth = '40px'

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label('g (m/s^2):'),
            dcc.Input(id='g-input', type='number', value=9.81, style={'width': inputWidth}),
            html.Label('alpha:'),
            dcc.Input(id='alpha-input', type='number', value=0.1, style={'width': inputWidth}),
            html.Label('mass (kg):'),
            dcc.Input(id='m-input', type='number', value=1, style={'width': inputWidth}),
            html.Label('n_steps:'),
            dcc.Input(id='n_steps-input', type='number', value=50, style={'width': inputWidth}),
            html.Label('t0 (s):'),
            dcc.Input(id='t0-input', type='number', value=0, style={'width': inputWidth}),
            html.Label('t1 (s):'),
            dcc.Input(id='t1-input', type='number', value=30, style={'width': inputWidth}),
            html.Label('x0 (m):'),
            dcc.Input(id='x0-input', type='number', value=0, style={'width': inputWidth}),
            html.Label('y0 (m):'),
            dcc.Input(id='y0-input', type='number', value=0, style={'width': inputWidth}),
            html.Label('vx0 (m/s):'),
            dcc.Input(id='vx0-input', type='number', value=50, style={'width': inputWidth}),
            html.Label('vy0:'),
            dcc.Input(id='vy0-input', type='number', value=50, style={'width': inputWidth}),
            html.Button('Calculate', id='calculate-button', n_clicks=0),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center', 'gap': '10px'}),
    ], style={'backgroundColor': 'navy', 'padding': '10px', 'color': 'white'}),
    html.Div([
        dcc.Graph(id='trajectory-plot', style={'height': '90vh'})
    ], style={'height': '95vh'})
], style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column'})

@app.callback(
    Output('trajectory-plot', 'figure'),
    [Input('calculate-button', 'n_clicks')],
    [State('g-input', 'value'),
     State('alpha-input', 'value'),
     State('m-input', 'value'),
     State('n_steps-input', 'value'),
     State('t0-input', 'value'),
     State('t1-input', 'value'),
     State('x0-input', 'value'),
     State('y0-input', 'value'),
     State('vx0-input', 'value'),
     State('vy0-input', 'value')]
)
def update_graph(n_clicks, g, alpha, m, n_steps, t0, t1, x0, y0, vx0, vy0):
    vertical_acceleration = lambda t, y, vy: -g - alpha * vy / m
    horizontal_acceleration = lambda t, x, vx: -alpha * vx / m

    y, dy = secondOrderRungeKutta(vertical_acceleration, t0, t1, y0, vy0, n_steps)
    x, dx = secondOrderRungeKutta(horizontal_acceleration, t0, t1, x0, vx0, n_steps)

    t = np.linspace(t0, t1, n_steps+1)

    df = pd.DataFrame({
        't': t,
        'x': x,
        'y': y,
        'v_x': dx,
        'v_y': dy
    })

    df = df.loc[df['y'] >= 0]

    trace = go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='lines+markers',
        name='Trajectory',
        hovertemplate='<b>Time</b>: %{customdata[0]:.2f}s<br>' +
                      '<b>X</b>: %{x:.2f} m<br>' +
                      '<b>Y</b>: %{y:.2f} m<br>' +
                      '<b>V_x</b>: %{customdata[1]:.2f} m/s<br>' +
                      '<b>V_y</b>: %{customdata[2]:.2f} m/s<extra></extra>',
        customdata=df[['t', 'v_x', 'v_y']].values
    )

    layout = go.Layout(
        xaxis={'title': 'X Position (m)'},
        yaxis={'title': 'Y Position (m)'},
        margin={'l': 50, 'r': 50, 't': 50, 'b': 50}
    )

    return {'data': [trace], 'layout': layout}

if __name__ == '__main__':
    app.run_server(debug=True)