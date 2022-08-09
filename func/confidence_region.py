# for solver.py in SOLVER
import math
import numpy as np
import torch
import plotly.graph_objs as go
import plotly.io as pio
import statistics

from TEDEouS.solver import grid_format_prepare
pio.renderers.default = "browser"


def get_rms(records):
    """
        Root-mean-square (rms)
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def confidence_region_print(u, cfg, param, u_main, prepared_grid_main, variance=0):
    # Additionally, comparison of solutions
    for k in range(len(u_main)):
        error_rmse = np.sqrt(np.mean((u.reshape(-1) - u_main[k].reshape(-1)) ** 2))
        print(f'RMSE = {error_rmse}')

    grid = grid_format_prepare(param, "mat")

    if u.ndim == 2:

        mean_arr = np.zeros((u_main.shape[1], u_main.shape[2]))
        var_arr = np.zeros((u_main.shape[1], u_main.shape[2]))
        s_g_arr = np.zeros((u_main.shape[1], u_main.shape[2]))  # population standard deviation of data.
        s_arr = np.zeros((u_main.shape[1], u_main.shape[2]))  # sample standard deviation of data

        for i in range(u_main.shape[1]):
            for j in range(u_main.shape[2]):
                mean_arr[i, j] = np.mean(u_main[:, i, j])
                var_arr[i, j] = np.var(u_main[:, i, j])
                s_arr[i, j] = statistics.stdev(u_main[:, i, j])

        mean_tens = torch.from_numpy(mean_arr)
        var_tens = torch.from_numpy(var_arr)
        s_g_arr = torch.from_numpy(var_arr) ** (1 / 2)
        s_arr = torch.from_numpy(s_arr)

        # Confidence region for the mean
        upper_bound = mean_tens + 1.96 * s_arr / math.sqrt(len(u_main))
        lower_bound = mean_tens - 1.96 * s_arr / math.sqrt(len(u_main))

        mean_tens = mean_tens.reshape(-1)
        upper_bound = upper_bound.reshape(-1)
        lower_bound = lower_bound.reshape(-1)
        # building 3-dimensional graph

        if cfg.params["glob_solver"]["mode"] == 'mat':
            fig = go.Figure(data=[
                go.Mesh3d(x=prepared_grid_main[0].reshape(-1), y=prepared_grid_main[1].reshape(-1), z=mean_tens, name='Solution field',
                          legendgroup='s', showlegend=True, color='lightpink',
                          opacity=1),
                go.Mesh3d(x=prepared_grid_main[0].reshape(-1), y=prepared_grid_main[1].reshape(-1), z=upper_bound, name='Confidence region',
                          legendgroup='c', showlegend=True, color='blue',
                          opacity=0.20),
                go.Mesh3d(x=prepared_grid_main[0].reshape(-1), y=prepared_grid_main[1].reshape(-1), z=lower_bound, name='Confidence region',
                          legendgroup='c', color='blue', opacity=0.20)
            ])

        else:
            fig = go.Figure(data=[
                go.Mesh3d(x=prepared_grid_main[:, 0], y=prepared_grid_main[:, 1], z=mean_tens, name='Solution field',
                          legendgroup='s', showlegend=True, color='lightpink',
                          opacity=1),
                go.Mesh3d(x=prepared_grid_main[:, 0], y=prepared_grid_main[:, 1], z=upper_bound, name='Confidence region',
                          legendgroup='c', showlegend=True, color='blue',
                          opacity=0.20),
                go.Mesh3d(x=prepared_grid_main[:, 0], y=prepared_grid_main[:, 1], z=lower_bound, name='Confidence region',
                          legendgroup='c', color='blue', opacity=0.20)
            ])

        fig.add_trace(go.Mesh3d(x=grid[0].reshape(-1), y=grid[1].reshape(-1), z=torch.from_numpy(u).reshape(-1),
                      name='Initial field',
                      legendgroup='i', showlegend=True, color='rgb(139,224,164)',
                      opacity=0.5))

        if variance:
            noise = []
            for i in range(u.shape[0]):
                noise.append(np.random.normal(0, variance * get_rms(u[i, :]), u.shape[1]))
            noise = np.array(noise)
            fig.add_trace(go.Mesh3d(x=grid[0].reshape(-1), y=grid[1].reshape(-1), z=torch.from_numpy(u + noise).reshape(-1),
                                    name='Initial field + noise',
                                    legendgroup='i_n', showlegend=True, color='rgb(139,224,80)',
                                    opacity=0.5))

        fig.update_layout(scene_aspectmode='auto')
        fig.update_layout(showlegend=True,
                          scene=dict(
                              xaxis_title='x1',
                              yaxis_title='x2',
                              zaxis_title='u',
                              aspectratio={"x": 1, "y": 1, "z": 1}
                          ),
                          height=800, width=800
                          )

        fig.show()

        # building Heatmap solution field and Heatmap the variance
        if cfg.params["glob_solver"]["mode"] == 'mat':
            fig = go.Figure(data=
                            go.Contour(x=prepared_grid_main[0].reshape(-1),
                                       y=prepared_grid_main[1].reshape(-1),
                                       z=mean_tens,
                                       contours_coloring='heatmap'))
            fig.update_layout(
                title_text='Visualization of the equation solution'
            )
            fig.show()

            fig = go.Figure(data=
                            go.Contour(x=prepared_grid_main[0].reshape(-1),
                                       y=prepared_grid_main[1].reshape(-1),
                                       z=var_tens.reshape(-1),
                                       contours_coloring='heatmap'))
            fig.update_layout(
                title_text='Visualization of the variance'
            )
            fig.show()
        else:
            fig = go.Figure(data=
                            go.Contour(x=prepared_grid_main[:, 0],
                                       y=prepared_grid_main[:, 1],
                                       z=mean_tens,
                                       contours_coloring='heatmap'))
            fig.update_layout(
                title_text='Visualization of the equation solution'
            )
            fig.show()

            fig = go.Figure(data=
                            go.Contour(x=prepared_grid_main[:, 0],
                                       y=prepared_grid_main[:, 1],
                                       z=var_tens.reshape(-1),
                                       contours_coloring='heatmap'))
            fig.update_layout(
                title_text='Visualization of the variance'
            )
            fig.show()
