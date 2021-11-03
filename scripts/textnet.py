### these should go easy
import sys
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_rows', 150)

import numpy as np
import os
import string
import collections
import math
import random
import statistics as stat
import re
import unicodedata
import json

# Natural Language Processing Toolkit - we use it especially for building bigrams
import nltk
from nltk.collocations import *

### for visualization
# in some cases I use matplotlib, which is much easier to configure, elsewhere I prefer Plotly, which is more "sexy"
import matplotlib.pyplot as plt
from PIL import Image

import seaborn as sns

# There is a lot of changes in Plotly nowadays. Perhaps some modifications of the code will be needed at some point
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
plotly.offline.init_notebook_mode(connected=True)

# gensim parts
from gensim import corpora
from gensim import models

### scikit-learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

### for network analysis
import networkx as nx


def construct_ego_network(source_network, term, num_of_neighbours, reduced=True):
    length, path = nx.single_source_dijkstra(source_network, term, target=None, weight="distance")
    shortest_nodes = list(length.keys())[0:num_of_neighbours+1]
    path_values_sorted = [dict_pair[1] for dict_pair in sorted(path.items(), key=lambda pair: list(length.keys()).index(pair[0]))]
    path_edges = []
    for path_to_term in path_values_sorted[1:num_of_neighbours+1]:
        path_edges.extend([tuple(sorted(bigram)) for bigram in nltk.bigrams(path_to_term)])
    shortest_edges = list(set(path_edges))
    ego_network = source_network.copy(as_view=False)
    nodes_to_remove = []
    for node in ego_network.nodes:
        if node not in shortest_nodes:
            nodes_to_remove.append(node)
    for element in nodes_to_remove:
        ego_network.remove_node(element)    
    edges_to_remove = []
    if reduced==True:
        for edge in ego_network.edges:
            if edge not in shortest_edges:
                if (edge[1],edge[0]) not in shortest_edges:
                    edges_to_remove.append(edge)
        for element in edges_to_remove:
            ego_network.remove_edge(element[0], element[1])
    return ego_network

def extract_ego_network_data(ego_network, term):
    ego_network_data_prec = sorted(list(ego_network.edges.data("weight")), key=lambda tup: int(tup[2]), reverse=True)
    ego_network_data_complete = []
    for tup in ego_network_data_prec:
      if tup[1] == term:
        ego_network_data_complete.append([tup[1], tup[0], int(tup[2]), round(1 / int(tup[2]), 5)])
      else:
        ego_network_data_complete.append([tup[0], tup[1], int(tup[2]), round(1 / int(tup[2]), 5)])
    return ego_network_data_complete

def get_nn(source_network, source_node, per_level=5):
    return [(source_node, n[0]) for n in sorted(source_network[source_node].items(), key=lambda edge: edge[1]['weight'], reverse=True)][:per_level]


def construct_association_network(source_network, source_term, per_level=5):
    assoc_edges = [] 
    assoc_edges.extend(get_nn(source_network, source_term, per_level))
    neighbors = [e[1] for e in assoc_edges]
    for nn in neighbors:
        assoc_edges.extend(get_nn(source_network, nn))

    assoc_network = source_network.copy(as_view=False)
    edges_to_remove = []
    for edge in assoc_network.edges():
        if edge not in assoc_edges:
            if (edge[1],edge[0]) not in assoc_edges:
                edges_to_remove.append(edge)
    assoc_network.remove_edges_from(edges_to_remove)
    isolates = nx.isolates(assoc_network)
    assoc_network.remove_nodes_from([n for n in isolates])
    return assoc_network



def network_from_lemmata(lemmata_list, weight_threshold=0.1):
    '''From a list of words'''
    #lemmata_list = [lemma for lemma in lemmata_list if lemma != "εἰμί"]
    bigrams_list = []
    for bigram in nltk.bigrams(lemmata_list):
        if bigram[0] != bigram[1]:
            bigrams_list.append(tuple(sorted(bigram)))
    G = network_object_from_bigrams(bigrams_list, weight_threshold)
    return G

def network_from_lemmata_lists(list_of_lists, weight_threshold=0.1):
    bigrams_list = []
    for lemmata_list in list_of_lists:
        for bigram in nltk.bigrams(lemmata_list):
            if bigram[0] != bigram[1]:
                bigrams_list.append(tuple(sorted(bigram)))
    G = network_object_from_bigrams(bigrams_list, weight_threshold)
    return G

def network_object_from_bigrams(bigrams_list, weight_threshold):
    try:
        bigrams_counts = list((collections.Counter(bigrams_list)).items())
        bigrams_counts = sorted(bigrams_counts, key=lambda x: x[1], reverse=True)
        G = nx.Graph()
        G.clear()
        G.add_weighted_edges_from(np.array([(bigram_count[0][0], bigram_count[0][1],  int(bigram_count[1])) for bigram_count in bigrams_counts]))
                ### add edges attributes 
        for (u, v, wt) in G.edges.data('weight'):
                G[u][v]["weight"] = int(wt)
        total_weight = sum([int(n) for n in nx.get_edge_attributes(G, "weight").values()])
        weights = sorted([int(n) for n in nx.get_edge_attributes(G, "weight").values()], reverse=True)
        index_position = int(len(weights) * weight_threshold)
        minimal_weight_value = 1
        edges_to_remove = []
        for edge in G.edges:
            if G[edge[0]][edge[1]]["weight"] < minimal_weight_value:
                    edges_to_remove.append(edge)
        for element in edges_to_remove:
            G.remove_edge(element[0], element[1])
        for (u, v) in G.edges:
            G[u][v]["norm_weight"] = G[u][v]["weight"] / total_weight
            G[u][v]["distance"] = round(1 / (G[u][v]["weight"]), 5)
            G[u][v]["norm_distance"] = round(1 / (G[u][v]["norm_weight"] ), 5)
    except: 
        G = nx.Graph()
        G.clear()
    return G

def network_from_sentences(sentences, weight_threshold=0.1):
    try:
        vocabulary =  list(set([word for sent in sentences for word in sent]))
        bow = CountVectorizer(vocabulary=vocabulary)
        bow_term2doc = bow.fit_transform([" ".join(doc) for doc in sentences]) ### run the model
        term2term_bow = (bow_term2doc.T * bow_term2doc)
        G = nx.from_numpy_matrix(term2term_bow.todense()) # from_pandas_adjacency()
        vocab_dict = dict(zip(range(len(vocabulary)), vocabulary))
        G = nx.relabel_nodes(G, vocab_dict)
        edges_to_remove = []
        for edge in G.edges:
            if edge[0] == edge[1]:
                edges_to_remove.append(edge)
        for element in edges_to_remove:
            G.remove_edge(element[0], element[1])
        total_weight = sum([int(n) for n in nx.get_edge_attributes(G, "weight").values()])
        weights = sorted([int(n) for n in nx.get_edge_attributes(G, "weight").values()], reverse=True)
        index_position = int(len(weights) * weight_threshold)
        minimal_weight_value = 2
        edges_to_remove = []
        for edge in G.edges:
            if G[edge[0]][edge[1]]["weight"] < 2:
                edges_to_remove.append(edge)
        for element in edges_to_remove:
            G.remove_edge(element[0], element[1])
        edges_to_remove = []
        for edge in G.edges:
            if edge[0] == edge[1]:
                edges_to_remove.append(edge)
        for element in edges_to_remove:
            G.remove_edge(element[0], element[1])
        for (u, v) in G.edges:
            G[u][v]["norm_weight"] = round((G[u][v]["weight"] / total_weight), 5)
            G[u][v]["distance"] = round(1 / (G[u][v]["weight"]), 5)
            G[u][v]["norm_distance"] = round(1 / (G[u][v]["norm_weight"] ), 5)
    except:
        G = nx.Graph()
        G.clear()
    return G

def draw_2d_network(networkx_object, width=500, height=500, fontsize=14):
    '''take networkX object and draw it'''
    pos_2d=nx.kamada_kawai_layout(networkx_object, weight="weight_norm")
    nx.set_node_attributes(networkx_object, pos_2d, "pos_2d")
    dmin=1
    ncenter=0
    Edges = list(networkx_object.edges)
    L=len(Edges)
    labels= list(networkx_object.nodes)
    N = len(labels)
    distance_list = [float(distance[2]) for distance in list(networkx_object.edges.data("distance"))]
    weight_list = [int(float(weight[2])) for weight in list(networkx_object.edges.data("weight"))]
    for n in pos_2d:
        x,y=pos_2d[n]
        d=(x-0.5)**2+(y-0.5)**2
        if d<dmin:
            ncenter=n
            dmin=d
    p =nx.single_source_shortest_path_length(networkx_object, ncenter)
    adjc= list(dict(networkx_object.degree()).values())
    middle_node_trace = go.Scatter(
        x=[],
        y=[],
        opacity=0,
        text=weight_list,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            opacity=0
            )
        )
    for Edge in Edges:
        x0,y0 = networkx_object.nodes[Edge[0]]["pos_2d"]
        x1,y1 = networkx_object.nodes[Edge[1]]["pos_2d"]
        middle_node_trace['x'] += tuple([(x0+x1)/2])
        middle_node_trace['y'] += tuple([(y0+y1)/2])
    edge_trace1 = go.Scatter(
        x=[], y=[],
        #hoverinfo='none',
        mode='lines',
        line=dict(width=4,color="#000000"),
        )
    edge_trace2 = go.Scatter(
        x=[],y=[],
        #hoverinfo='none',
        mode='lines',
        line=dict(width=2,color="#404040"),
        )
    edge_trace3 = go.Scatter(
        x=[], y=[],
        #hoverinfo='none',
        mode='lines',
        line=dict(width=1,color="#C0C0C0"),
        )
    best_5percent_norm_weight = sorted(list(networkx_object.edges.data("norm_weight")), key=lambda x: x[2], reverse=True)[int((len(networkx_object.edges.data("norm_weight")) / 100) * 5)][2]
    best_20percent_norm_weight = sorted(list(networkx_object.edges.data("norm_weight")), key=lambda x: x[2], reverse=True)[int((len(networkx_object.edges.data("norm_weight")) / 100) * 20)][2]
    for edge in networkx_object.edges.data():
        if edge[2]["norm_weight"] >= best_5percent_norm_weight:
            x0, y0 = networkx_object.nodes[edge[0]]['pos_2d']
            x1, y1 = networkx_object.nodes[edge[1]]['pos_2d']
            edge_trace1['x'] += tuple([x0, x1, None])
            edge_trace1['y'] += tuple([y0, y1, None])
        else:
            if edge[2]["norm_weight"] >= best_20percent_norm_weight:
                x0, y0 = networkx_object.nodes[edge[0]]['pos_2d']
                x1, y1 = networkx_object.nodes[edge[1]]['pos_2d']
                edge_trace2['x'] += tuple([x0, x1, None])
                edge_trace2['y'] += tuple([y0, y1, None])
            else:
                x0, y0 = networkx_object.nodes[edge[0]]['pos_2d']
                x1, y1 = networkx_object.nodes[edge[1]]['pos_2d']
                edge_trace3['x'] += tuple([x0, x1, None])
                edge_trace3['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        #name=[],
        text=[],
        textposition='bottom center',
        textfont_size=fontsize,
        mode='markers+text',
        hovertext=adjc,
        hoverinfo='text',
        marker=dict(
            ###showscale=True,
            showscale=False, ### change to see scale
            colorscale='Greys',
            #reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=30,
                title='degree',
                xanchor='left',
                titleside='right'
                ),
            line=dict(width=1.5)
            )
        )

    for node in networkx_object.nodes():
        x, y = networkx_object.nodes[node]['pos_2d']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace["text"] += tuple([node])
        ### original version: node_trace["text"] += tuple([node])

    ### Color Node Points
    for node, adjacencies in enumerate(nx.generate_adjlist(networkx_object)):
        node_trace['marker']['color'] += tuple([len(adjacencies)])
        ###node_info = ' of connections: '+str(len(adjacencies))
        ###node_trace['something'].append(node_info)

    fig = go.Figure(data=[edge_trace1, edge_trace2, edge_trace3, node_trace, middle_node_trace],
        layout=go.Layout(
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=False,
            width=width,
            height=height,
            #title=file_name,
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=10,l=10,r=10, t=10),
            xaxis=dict(range=[-1.15, 1.05], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-1.15, 1.05], showgrid=False, zeroline=False, showticklabels=False)
            ))
    return fig

def draw_3d_network(networkx_object):
    '''take networkX object and draw it in 3D'''
    Edges = list(networkx_object.edges)
    L=len(Edges)
    distance_list = [distance[2] for distance in list(networkx_object.edges.data("distance"))]
    weight_list = [int(float(weight[2])) for weight in list(networkx_object.edges.data("weight"))]
    labels= list(networkx_object.nodes)
    N = len(labels)
    adjc= list(dict(networkx_object.degree()).values())
    pos_3d=nx.spring_layout(networkx_object, weight="weight", dim=3)
    nx.set_node_attributes(networkx_object, pos_3d, "pos_3d")
    layt = [list(array) for array in pos_3d.values()]
    N= len(networkx_object.nodes)
    Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[layt[k][1] for k in range(N)]# y-coordinates
    Zn=[layt[k][2] for k in range(N)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for Edge in Edges:
        Xe+=[networkx_object.nodes[Edge[0]]["pos_3d"][0],networkx_object.nodes[Edge[1]]["pos_3d"][0], None]# x-coordinates of edge ends
        Ye+=[networkx_object.nodes[Edge[0]]["pos_3d"][1],networkx_object.nodes[Edge[1]]["pos_3d"][1], None]
        Ze+=[networkx_object.nodes[Edge[0]]["pos_3d"][2],networkx_object.nodes[Edge[1]]["pos_3d"][2], None]

        ### to get the hover into the middle of the line
        ### we have to produce a node in the middle of the line
        ### based on https://stackoverflow.com/questions/46037897/line-hover-text-in-plotly

    middle_node_trace = go.Scatter3d(
            x=[],
            y=[],
            z=[],
            opacity=0,
            text=weight_list,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                opacity=0
            )
        )

    for Edge in Edges:
        x0,y0,z0 = networkx_object.nodes[Edge[0]]["pos_3d"]
        x1,y1,z1 = networkx_object.nodes[Edge[1]]["pos_3d"]
        ###trace3['x'] += [x0, x1, None]
        ###trace3['y'] += [y0, y1, None]
        ###trace3['z'] += [z0, z1, None]
        ###trace3_list.append(trace3)
        middle_node_trace['x'] += tuple([(x0+x1)/2])
        middle_node_trace['y'] += tuple([(y0+y1)/2])#.append((y0+y1)/2)
        middle_node_trace['z'] += tuple([(z0+z1)/2])#.append((z0+z1)/2)
        
    ### edge trace
    trace1=go.Scatter3d(x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       line=dict(color='rgb(125,125,125)', width=1),
                       text=distance_list,
                       hoverinfo='text',
                       textposition="top right"
                       )
    ### node trace
    trace2=go.Scatter3d(x=Xn,
                       y=Yn,
                       z=Zn,
                       mode='markers+text',
                       ###name=labels,
                       marker=dict(symbol='circle',
                                     size=6,
                                     color=adjc,
                                     colorscale='Earth',
                                     reversescale=True,
                                     line=dict(color='rgb(50,50,50)', width=0.5)
                                     ),
                       text=[],
                       #textposition='bottom center',
                       #hovertext=adjc,
                       #hoverinfo='text'
                       )
    for node in networkx_object.nodes():
        trace2["text"] += tuple([node])
    
    axis=dict(showbackground=False,
                  showline=False,
                  zeroline=False,
                  showgrid=False,
                  showticklabels=False,
                  title=''
                  )
    layout = go.Layout(
                plot_bgcolor='rgba(0,0,0,0)',
                 title="",
                 width=900,
                 height=700,
                 showlegend=False,
                 scene=dict(
                     xaxis=dict(axis),
                     yaxis=dict(axis),
                     zaxis=dict(axis),
                ),
             margin=dict(
                t=100
            ),
            hovermode='closest',
            annotations=[
                   dict(
                   showarrow=False,
                    text="",
                    xref='paper',
                    yref='paper',
                    x=0,
                    y=0.1,
                    xanchor='left',
                    yanchor='bottom',
                    font=dict(
                    size=14
                    )
                    )
                ],    )
    data=[trace1, trace2, middle_node_trace]
    fig=go.Figure(data=data, layout=layout)
    return fig