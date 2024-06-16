import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import pickle as pk
import numpy as np
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='TrackFittingFinetuning',
                    description='Fine tunes a model for Track Fitting in SFG',)

    parser.add_argument('-p', '--per_particle', action='store_true', help='analysis per particle instead of per event')
    args = parser.parse_args()
    
    with open("/scratch4/maubin/results/vatransformer_v5_analysis.pk","rb") as f:
        (per_event_analysis,per_particle_analysis)=pk.load(f)

    if args.per_particle:
        df_pandas = pd.DataFrame(per_particle_analysis)
    else:
        df_pandas = pd.DataFrame(per_event_analysis)


    # Create a list of columns for dropdown options
    columns = df_pandas.columns.tolist()

    # Create Dash application
    app = Dash(__name__)

    app.layout = html.Div([
        # html.H1("Interactive Distribution Plot with Variable Selection and Dark Mode", style={'color': 'white'}),
        
        # Main container
        html.Div([
            # Plot area (left side, two-thirds)
            html.Div([
                dcc.Graph(id='distribution-plot', style={'height': '100vh'})
            ], style={'width': '66%', 'display': 'inline-block', 'vertical-align': 'top'}),
            
            html.Div([
                # Controls (right side, one-third)
                html.Div([
                    html.Label("Number of Bins for Histogram:", style={'color': 'white'}),
                        dcc.Input(
                            id='num-bins',
                            type='number',
                            value=100,
                            min=1,
                            style={'backgroundColor': '#444', 'color': 'white', 'borderColor': '#555'}
                        ),
                ], style={'padding': 10}),
                html.Div([
                    html.Label("Select Muon ke range:", style={'color': 'white'}),
                    dcc.RangeSlider(
                        id='y-filter',
                        min=df_pandas['muon_ke'].min(),
                        max=df_pandas['muon_ke'].max(),
                        step=10,
                        value=[df_pandas['muon_ke'].min(), df_pandas['muon_ke'].max()],
                        marks={1000*i: str(i)+" GeV" for i in range(int(df_pandas['muon_ke'].min()//1000), int(df_pandas['muon_ke'].max()//1000) + 1)}
                    ),
                ], style={'padding': 10}),

                # Z Filter
                html.Div([
                    html.Label("Select ke range:" if args.per_particle else "Select Total ke range:", style={'color': 'white'}),
                    dcc.RangeSlider(
                        id='z-filter',
                        min=df_pandas['ke' if args.per_particle else 'total_ke'].min(),
                        max=df_pandas['ke' if args.per_particle else 'total_ke'].max(),
                        step=1,
                        value=[df_pandas['ke' if args.per_particle else 'total_ke'].min(), df_pandas['ke' if args.per_particle else 'total_ke'].max()],
                        marks= {10*i: str(10*i)+" MeV" for i in range(int(df_pandas['ke'].min()//10), int(df_pandas['ke'].max()//10) + 1)} if args.per_particle else {100*i: str(100*i)+" MeV" for i in range(int(df_pandas['total_ke'].min()//100), int(df_pandas['total_ke'].max()//100) + 1)}
                    ),
                ], style={'padding': 10}),
                
                # W filter
                html.Div([
                    dcc.Dropdown(
                        id='w-filter',
                        options=[{'label': val, 'value': val} for val in np.sort(df_pandas['nb_part'].unique())],
                        multi=True,
                        placeholder="Select number of additional particles to filter",
                        style={'backgroundColor': '#444', 'color': 'black', 'borderColor': '#555'}
                    ),
                ], style={'padding': 10}),
                
                # V filter
                html.Div([
                    dcc.Dropdown(
                        id='v-filter',
                        options=[{'label': val, 'value': val} for val in np.sort(df_pandas['nb_protons'].unique())],
                        multi=True,
                        placeholder="Select number of additional protons to filter",
                        style={'backgroundColor': '#444', 'color': 'black', 'borderColor': '#555'}
                    ),
                ], style={'padding': 10}),
                
                # U filter
                html.Div([
                    dcc.Dropdown(
                        id='u-filter',
                        options=[{'label': val, 'value': val} for val in np.sort(df_pandas['pid' if args.per_particle else 'nb_neutrons'].unique())],
                        multi=True,
                        placeholder="Select pid to filter" if args.per_particle else "Select number of additional neutrons to filter",
                        style={'backgroundColor': '#444', 'color': 'black', 'borderColor': '#555'}
                    ),
                ], style={'padding': 10}),


                # Variable Selection for Subplots
                html.Div([
                    html.Label("Select Variable for the histogram:", style={'color': 'white'}),
                    dcc.Dropdown(
                        id='var1',
                        options=[{'label': col, 'value': col} for col in columns if col not in ['Y', 'Z']],
                        value='X1',
                        style={'backgroundColor': '#444', 'borderColor': '#555'}
                    ),
                    html.Label("Select Variable for the line plot:", style={'color': 'white'}),
                    dcc.Dropdown(
                        id='var1b',
                        options=[{'label': col, 'value': col} for col in columns if col not in ['Y', 'Z']],
                        value='X1b',
                        style={'backgroundColor': '#444', 'borderColor': '#555'}
                    ),
                    # html.Label("Select Variable for Subplot 2:", style={'color': 'white'}),
                    # dcc.Dropdown(
                    #     id='var2',
                    #     options=[{'label': col, 'value': col} for col in columns if col not in ['Y', 'Z']],
                    #     value='X2'
                    # ),
                    # html.Label("Select Variable for Subplot 3:", style={'color': 'white'}),
                    # dcc.Dropdown(
                    #     id='var3',
                    #     options=[{'label': col, 'value': col} for col in columns if col not in ['Y', 'Z']],
                    #     value='X3'
                    # ),
                ], style={'padding': 10, 'display': 'flex', 'flexDirection': 'column', 'gap': '10px'})
            ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top', 'backgroundColor': '#333'})
        ])
    ], style={'backgroundColor': '#222', 'padding': '10px'})

    @app.callback(
        Output('distribution-plot', 'figure'),
        [
            Input('num-bins', 'value'),
            Input('y-filter', 'value'),
            Input('z-filter', 'value'),
            Input('w-filter', 'value'),
            Input('v-filter', 'value'),
            Input('u-filter', 'value'),
            Input('var1', 'value'),
            Input('var1b', 'value'),
            # Input('var2', 'value'),
            # Input('var3', 'value')
        ]
    )
    def update_plot(num_bins,
                    y_range, z_range, 
                    w_values, v_values, u_values,
                    var1, var1b,
                    # var2, var3,
                    ):
        # Filter data based on selected Y and Z range
        min_y, max_y = y_range
        min_z, max_z = z_range
        
        filtered_df = df_pandas[
            (df_pandas['muon_ke'] >= min_y) & (df_pandas['muon_ke'] <= max_y) &
            (df_pandas['ke' if args.per_particle else 'total_ke'] >= min_z) & (df_pandas['ke' if args.per_particle else 'total_ke'] <= max_z)
        ]
        
        if w_values:
            filtered_df = filtered_df[filtered_df['nb_part'].isin(w_values)]
        else:
            filtered_df = filtered_df
            
            
        if v_values:
            filtered_df = filtered_df[filtered_df['nb_protons'].isin(v_values)]
        else:
            filtered_df = filtered_df
            
            
        if u_values:
            filtered_df = filtered_df[filtered_df['pid' if args.per_particle else 'nb_neutrons'].isin(u_values)]
        else:
            filtered_df = filtered_df
        
        # Create plot
        fig = go.Figure()
        
        if var1:
            
            
            # Function to calculate statistics and generate annotations
            def get_statistics(df_column):
                mean_val = df_column.mean()
                median_val = df_column.median()
                std_val = df_column.std()
                return mean_val, median_val, std_val

            def add_annotations(fig, stats, row):
                mean_val, median_val, std_val = stats
                annotations = [
                    dict(
                        xref=f'x{row}', yref=f'y{row}',
                        x=0.5, y=0.95, xanchor='left', yanchor='top',
                        text=f"<b>Mean:</b> {mean_val:.2f}, <b>Median:</b> {median_val:.2f}, <b>Std:</b> {std_val:.2f}",
                        showarrow=False,
                        font=dict(size=12, color="white")
                    )
                ]
                fig.add_annotation(annotations[0])

            # Add histogram and statistics for var1
            fig.add_trace(
                go.Histogram(x=filtered_df[var1], nbinsx=num_bins, name=var1, marker_color='blue'),
                # row=1, col=1
            )
            stats_var1 = get_statistics(filtered_df[var1])
            add_annotations(fig, stats_var1, 1)
            
            # Update layout to add axis labels and adjust layout
            fig.update_layout(
                title=f'Distribution of {var1}',
                xaxis_title=var1,
                yaxis=dict(
                    title=f'{var1} Distribution',
                    titlefont=dict(color='blue'),
                    tickfont=dict(color='blue')
                ),
            )
            
            if var1b:
                # Determine the bins for X1
                bins = np.histogram_bin_edges(filtered_df[var1], bins=num_bins)
                
                # Digitize the X1 values to bin them
                binned_x1 = np.digitize(filtered_df[var1], bins) - 1

                # Create a DataFrame to store the stats
                binned_stats = pd.DataFrame({
                    'bin': binned_x1,
                    'X1b': filtered_df[var1b]
                })
        
                # Calculate mean and standard deviation of X2 for each bin
                bin_means = binned_stats.groupby('bin')['X1b'].mean()
                bin_stds = binned_stats.groupby('bin')['X1b'].std()

                # Calculate bin centers for the x-axis
                bin_centers = (bins[:-1] + bins[1:]) / 2
                
                # Filter out bins that are not represented in the data
                bin_centers = bin_centers[:len(bin_means)]
                
                # Add mean and standard deviation for var2 per bin
                fig.add_trace(
                    go.Scatter(
                        x=bin_centers, 
                        y=bin_means, 
                        error_y=dict(type='data', array=bin_stds),
                        name=f'{var1b} Mean ± Std Dev',
                        marker_color='orange',
                        yaxis='y2',
                        mode='lines+markers'
                    )
                )
                
                # Update layout to add axis labels and adjust layout
                fig.update_layout(
                    title=f'Distribution of {var1} and Mean ± Std Dev of {var1b}',
                    yaxis2=dict(
                        title=f'{var1b} Mean ± Std Dev',
                        titlefont=dict(color='orange'),
                        tickfont=dict(color='orange'),
                        overlaying='y',
                        side='right'
                    ),
                )
            
            
            # # Update individual x-axis labels for subplots
            # fig.update_xaxes(title_text=var1, row=1, col=1)
            
            # # Update individual y-axis labels for subplots
            # fig.update_yaxes(title_text="Distribution", row=1, col=1)

        # # Add histogram and statistics for var2
        # fig.add_trace(
        #     go.Histogram(x=filtered_df[var2], nbinsx=100, name=var2, marker_color='green'),
        #     row=2, col=1
        # )
        # stats_var2 = get_statistics(filtered_df[var2])
        # add_annotations(fig, stats_var2, 2)

        # # Add histogram and statistics for var3
        # fig.add_trace(
        #     go.Histogram(x=filtered_df[var3], nbinsx=100, name=var3, marker_color='red'),
        #     row=3, col=1
        # )
        # stats_var3 = get_statistics(filtered_df[var3])
        # add_annotations(fig, stats_var3, 3)

        # Update layout to add axis labels and adjust layout
        fig.update_layout(
            # title_text=f'Distributions for Muon ke in range [{min_y}, {max_y}] and Total ke in range [{min_z}, {max_z}]',
            # title_text=f'Distributions of variables',
            # height=1200,  # Adjust the height for better visualization
            # width = 1000,
            showlegend=False,
            paper_bgcolor='#222',  # Dark background for plot
            plot_bgcolor='#333',  # Dark background for subplots
            font=dict(color='white')  # White font for text
        )

        # Update individual x-axis labels for subplots
        # fig.update_xaxes(title_text=var2, row=2, col=1)
        # fig.update_xaxes(title_text=var3, row=3, col=1)

        # fig.update_yaxes(title_text="Distribution", row=2, col=1)
        # fig.update_yaxes(title_text="Distribution", row=3, col=1)

        return fig
    app.run_server(debug=True, host='0.0.0.0', port=8050)  # Bind to all IPs on the remote cluster
