# for solver.py in SOLVER
import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio

pio.renderers.default = "browser"


def confidence_region_print(u_main, prepared_grid_main, title=None):
    mean_arr = np.zeros((u_main.shape[1], u_main.shape[2]))
    var_arr = np.zeros((u_main.shape[1], u_main.shape[2]))

    for i in range(u_main.shape[1]):
        for j in range(u_main.shape[2]):
            mean_arr[i, j] = np.mean(u_main[:, i, j])
            var_arr[i, j] = np.var(u_main[:, i, j])

    mean_tens = torch.from_numpy(mean_arr)
    var_tens = torch.from_numpy(var_arr)

    upper_bound = mean_tens + 1.96 * var_tens
    lower_bound = mean_tens - 1.96 * var_tens

    mean_tens = mean_tens.reshape(-1)
    upper_bound = upper_bound.reshape(-1)
    lower_bound = lower_bound.reshape(-1)

    if prepared_grid_main.shape[1] == 2:
        # building 3-dimensional graph
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

        fig.update_layout(scene_aspectmode='auto')
        fig.update_layout(showlegend=True,
                          scene=dict(
                              xaxis_title='x1',
                              yaxis_title='x2',
                              zaxis_title='u',
                              zaxis=dict(nticks=10, dtick=1),
                              aspectratio={"x": 1, "y": 1, "z": 1}
                          ),
                          height=800, width=800
                          )

        fig.show()

        # building Heatmap solution field
        fig = go.Figure(data=
                        go.Contour(x=prepared_grid_main[:, 0],
                                   y=prepared_grid_main[:, 1],
                                   z=mean_tens,
                                   contours_coloring='heatmap'))
        fig.update_layout(
            title_text='Visualization of the equation solution'
        )
        fig.show()

        # building Heatmap the variance
        fig = go.Figure(data=
                        go.Contour(x=prepared_grid_main[:, 0],
                                   y=prepared_grid_main[:, 1],
                                   z=var_tens.reshape(-1),
                                   contours_coloring='heatmap'))
        fig.update_layout(
            title_text='Visualization of the variance'
        )
        fig.show()

    # # plot with matplotlib
    # if prepared_grid_main.shape[1] == 2:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot_trisurf(prepared_grid_main[:, 0].reshape(-1), prepared_grid_main[:, 1].reshape(-1),
    #                     mean_tens, linewidth=0.2, alpha=1, label='mean')
    #     ax.plot_trisurf(prepared_grid_main[:, 0].reshape(-1), prepared_grid_main[:, 1].reshape(-1),
    #                     upper_bound, linewidth=0.2, alpha=0.5, label='upper_bound')
    #     ax.plot_trisurf(prepared_grid_main[:, 0].reshape(-1), prepared_grid_main[:, 1].reshape(-1),
    #                     lower_bound, linewidth=0.2, alpha=0.5, label='lower_bound')
    #     ax.set_xlabel("x1")
    #     ax.set_ylabel("x2")
    #     plt.show()
