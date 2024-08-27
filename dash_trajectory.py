import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from RungeKutta import secondOrderRungeKutta
from scipy.optimize import newton

g = 9.81

def exactTrajectory(x0:float, v0:float, a:float, alpha:float, m:float, t:np.array) -> np.array:
    """
    Exact solution for the trajectory of a particule of mass m, subject to a constant force m*a and  a drag force -alpha * v
    With initional condition x0 for position and v0 for speed
    """
    m_over_alpha = m / alpha
    coeff1 = a*m_over_alpha
    coeff2 = -m_over_alpha*(v0-coeff1)
    return coeff2*np.exp(-t/m_over_alpha)+np.add(coeff1*t, x0-coeff2)

app = dash.Dash(__name__, title='2D Trajectory with Drag')

inputWidth = '40px'

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label('alpha:'),
            dcc.Input(id='alpha-input', type='number', value=0.1, style={'width': inputWidth}),
            html.Label('mass (kg):'),
            dcc.Input(id='m-input', type='number', value=1, style={'width': inputWidth}),
            html.Label('n_steps:'),
            dcc.Input(id='n_steps-input', type='number', value=50, style={'width': inputWidth}),
            html.Label('x0 (m):'),
            dcc.Input(id='x0-input', type='number', value=0, style={'width': inputWidth}),
            html.Label('y0 (m):'),
            dcc.Input(id='y0-input', type='number', value=0, style={'width': inputWidth}),
            html.Label('vx0 (m/s):'),
            dcc.Input(id='vx0-input', type='number', value=50, style={'width': inputWidth}),
            html.Label('vy0:'),
            dcc.Input(id='vy0-input', type='number', value=50, style={'width': inputWidth}),
            html.Button('Calculate', id='calculate-button', n_clicks=0),
            dcc.RadioItems(
                id='graph-type',
                options=[
                    {'label': 'y(x)', 'value': 'y_x'},
                    {'label': 'x(t), vx(t), y(t), vy(t)', 'value': 'x_y_v'}
                ],
                value='y_x',
                labelStyle={'display': 'inline-block', 'marginLeft': '10px'}
            ),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center', 'gap': '10px'}),
    ], style={'backgroundColor': 'navy', 'padding': '10px', 'color': 'white'}),
    html.Div([
        dcc.Graph(id='trajectory-plot', style={'height': '90vh'})
    ], style={'height': '95vh'}),
    dcc.Store(id='trajectory-data')
], style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column'})

@app.callback(
    Output('trajectory-data', 'data'),
    [Input('calculate-button', 'n_clicks')],
    [State('alpha-input', 'value'),
     State('m-input', 'value'),
     State('n_steps-input', 'value'),
     State('x0-input', 'value'),
     State('y0-input', 'value'),
     State('vx0-input', 'value'),
     State('vy0-input', 'value')]
)
def compute_trajectory(n_clicks, alpha, m, n_steps, x0, y0, vx0, vy0):
    vertical_acceleration = lambda t, y, vy: -g - alpha * vy / m
    horizontal_acceleration = lambda t, x, vx: -alpha * vx / m

    tau = newton(lambda tau: exactTrajectory(y0, vy0, -g, alpha, m, t=tau) - 0, 10)
    t0 = 0
    y, dy = secondOrderRungeKutta(vertical_acceleration, t0, t0 + tau, y0, vy0, n_steps)
    x, dx = secondOrderRungeKutta(horizontal_acceleration, t0, t0 + tau, x0, vx0, n_steps)

    t = np.linspace(t0, t0+tau, n_steps+1)

    df = pd.DataFrame({
        't': t,
        'x': x,
        'y': y,
        'v_x': dx,
        'v_y': dy
    })

    return df.to_dict('records')

@app.callback(
    Output('trajectory-plot', 'figure'),
    [Input('trajectory-data', 'data'),
     Input('graph-type', 'value')]
)
def update_graph(data, graph_type):
    if not data:
        return go.Figure()

    df = pd.DataFrame(data)

    if graph_type == 'y_x':
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
            title=f"Trajectory time: {df['t'].iat[-1]:,.2f}s, horizontal distance covered: {df['x'].iat[-1]:,.1f}m",
            xaxis={'title': 'X Position (m)'},
            yaxis={'title': 'Y Position (m)'},
            margin={'l': 50, 'r': 50, 't': 50, 'b': 50}
        )

        return {'data': [trace], 'layout': layout}

    elif graph_type == 'x_y_v':
        fig = make_subplots(rows=2, cols=2, 
                            subplot_titles=('x(t)', 'y(t)', 'vx(t)', 'vy(t)'),
                            vertical_spacing=0.1,
                            horizontal_spacing=0.05)

        fig.add_trace(go.Scatter(x=df['t'], y=df['x'], name='x(t)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['t'], y=df['y'], name='y(t)'), row=1, col=2)
        fig.add_trace(go.Scatter(x=df['t'], y=df['v_x'], name='vx(t)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['t'], y=df['v_y'], name='vy(t)'), row=2, col=2)

        fig.update_xaxes(title_text='Time (s)', row=2, col=1)
        fig.update_xaxes(title_text='Time (s)', row=2, col=2)
        fig.update_yaxes(title_text='X Position (m)', row=1, col=1)
        fig.update_yaxes(title_text='Y Position (m)', row=1, col=2)
        fig.update_yaxes(title_text='X Velocity (m/s)', row=2, col=1)
        fig.update_yaxes(title_text='Y Velocity (m/s)', row=2, col=2)

        fig.update_layout(
            height=800,
            title_text=f"Trajectory time: {df['t'].iat[-1]:,.2f}s",
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )

        return fig

if __name__ == '__main__':
    app.run_server(debug=True)