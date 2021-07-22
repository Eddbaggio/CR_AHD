# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 14:56:50 2021

@author: gabri
"""
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd

if __name__ == '__main__':
    G = ox.graph_from_place('Innere Stadt, Vienna, Austria', network_type='all', simplify=True)
    fig, ax = ox.plot_graph(G, show=False, close=False)

    # read_csv function which is used to read the required CSV file
    data = pd.read_csv('New_1st.csv')

    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    # print(data)
    # new data frame with split value columns
    new = data["SHAPE"].str.split(" ", n=1, expand=True)

    # making separate first name column from new data frame
    data["LONGITUDE"] = new[0]

    # making separate last name column from new data frame
    data["LATITUDE"] = new[1]

    data.drop(columns=["SHAPE"], inplace=True)

    data['LATITUDE'] = pd.to_numeric(data['LATITUDE'])
    data['LONGITUDE'] = pd.to_numeric(data['LONGITUDE'])
    # df display
    print(data)
    data: pd.DataFrame

    # sample 100 random locations
    data = data.sample(100)

    # distance_matrix = []
    # # quadratic complexity -> this may take a while
    # for origin_idx, origin_xy in data.iterrows():
    #     distance_array = []
    #     origin_node = ox.distance.nearest_nodes(G, origin_xy['LONGITUDE'], origin_xy['LATITUDE'])
    #
    #     for destination_idx, destination_xy in data.iterrows():
    #         destination_node = ox.distance.nearest_nodes(G, destination_xy['LONGITUDE'], destination_xy['LATITUDE'])
    #
    #         distance = nx.shortest_path_length(G=G, source=origin_node, target=destination_node, weight='length')
    #         distance_array.append(distance)
    #
    #     distance_matrix.append(distance_array)
    #
    # print(distance_matrix)

    ax.scatter(data['LONGITUDE'], data['LATITUDE'], c='red')
    plt.show()
    pass
