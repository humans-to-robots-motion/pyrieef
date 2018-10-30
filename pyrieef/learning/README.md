This module allows to create 2D environments and learn how to
perform motion planning from data.

    python random_environment.py --numdatasets 200 # generates the environments quite fast
    python demonstrations.py # generates the trajectories, quite slow (dijkstra + newton traj-opt)
    python visualize_data.py --trajectories # see the end result (only smooth trajectories)

You can also see what the difference between the smooth and graph trajectories with the following

    python demonstrations.py --show_result
