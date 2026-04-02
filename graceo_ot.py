#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic Librariess
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import igraph as ig
from igraph import Graph
from itertools import product
import os

from math import radians, sin, cos, sqrt, atan2, asin
import plotly.graph_objects as go

from PAMI.extras.stats import graphDatabase as alg
from PAMI.subgraphMining.basic import gspan as alg
from PAMI.extras.visualize import graphs as vis

import overpy
from shapely.geometry import Polygon
import folium
from folium import plugins
import shapely.geometry
import scienceplots
import matplotlib
matplotlib.use('Agg')
import statistics
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import MonthLocator, DateFormatter
import string


### Libraries for predicting time series values
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

### Other libraries
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import plotly.express as px
import h3

from functools import reduce
from typing import List, Dict, Tuple

########################################

def read_files(path_vertices,path_edges, fields_nodes, fields_edges):
    vertices = pd.read_csv(path_vertices,sep=";", usecols = fields_nodes)
    edges = pd.read_csv(path_edges,sep="\t", usecols = fields_edges) # change with "\t" for seasons graphs (instead ";")
    edges = edges.rename(columns={"gid_from":"source","gid_to":"target"})
    edges_filter = edges[edges["NbPerMaxDurationDays_14"] >= 1]
    return vertices, edges_filter

def building_simple_graph_ig(vertices, edges):
    g = ig.Graph.DictList(
        vertices = vertices.to_dict('records'),
        edges = edges.to_dict('records'),
        directed = True,
        vertex_name_attr='id',
        #vertex_name_attr=vertices.index.values, 
        edge_foreign_keys=('source', 'target'));
    g.simplify(multiple=True, loops=True, combine_edges=None)
    return g


def build_graphs_by_time_old(df_vertices, df_edges, mode="month"):
    if mode not in ["month", "year"]:
        raise ValueError("Mode must be 'month' or 'year'")

    # Determine the grouping keys
    group_keys = ['year', 'month'] if mode == 'month' else ['year']

    # Group edges by time
    grouped = df_edges.groupby(group_keys)
    graph_list = []

    # Identify vertex ID column (assume it's the first column)
    vertex_id_col = df_vertices.columns[0]
    df_vertices = df_vertices.copy()
    df_vertices[vertex_id_col] = df_vertices[vertex_id_col].astype(str)

    for group_values, df_group in grouped:
        # Aggregate numeric edge attributes by (source, target) if mode is 'year'
        if mode == "year":
            df_group = df_group.groupby(['source', 'target'], as_index=False).sum(numeric_only=True)

        # Ensure sources and targets are strings (igraph uses string names)
        df_group['source'] = df_group['source'].astype(str)
        df_group['target'] = df_group['target'].astype(str)

        # Build edge list
        edge_list = list(zip(df_group["source"], df_group["target"]))

        # Create the graph
        g = Graph.TupleList(edge_list, directed=True)

        # Subset vertex dataframe to only include vertices in this graph
        graph_vertex_names = set(g.vs["name"])
        df_sub_vertices = df_vertices[df_vertices[vertex_id_col].isin(graph_vertex_names)]

        # Assign all vertex attributes
        for attr in df_vertices.columns:
            if attr == vertex_id_col:
                continue  # already used for naming
            attr_dict = df_sub_vertices.set_index(vertex_id_col)[attr].to_dict()
            g.vs[attr] = [attr_dict.get(v["name"], None) for v in g.vs]

        # Assign edge attributes (skip source/target/time columns)
        for attr in df_group.columns:
            if attr not in ['source', 'target', 'year', 'month']:
                g.es[attr] = df_group[attr].tolist()

        # Store time metadata in the graph
        if mode == "month":
            g["year"] = group_values[0]
            g["month"] = group_values[1]
        else:
            g["year"] = group_values
        print(g.vcount())
        graph_list.append(g)

    return graph_list

def build_graphs_by_time(df_vertices, df_edges, likes, mode="month"):
    
    graphs = []
    time_keys = []

    # Determine unique time keys (year or year-month)
    if mode == "month":
        time_keys = df_edges[['year', 'month']].drop_duplicates().sort_values(['year', 'month']).values.tolist()
    elif mode == "year":
        time_keys = df_edges[['year']].drop_duplicates().sort_values(['year']).values.tolist()

    for key in time_keys:
        if mode == "month":
            year, month = key
            df_edges_filtered = df_edges[(df_edges['year'] == year) & (df_edges['month'] == month)]
            df_likes_filtered = likes[(likes['year'] == year) & (likes['month'] == month)]
        else:
            year = key[0]
            df_edges_filtered = df_edges[df_edges['year'] == year]
            df_likes_filtered = likes[likes['year'] == year].groupby('id')[['rating', 'NB']].mean().reset_index()

        # Create empty graph
        g = Graph()
        g.add_vertices(df_vertices.shape[0])

        # Set node attributes from df_vertices
        for col in df_vertices.columns:
            g.vs[col] = df_vertices[col].tolist()

        # Add node index mapping by idplace
        idplace_to_index = dict(zip(df_vertices['id'], range(df_vertices.shape[0])))

        # Merge likes info to node attributes
        likes_map = df_likes_filtered.set_index('id')[['rating', 'NB']].to_dict('index')

        # Initialize default values
        ratings = []
        nbs = []
        for v in g.vs:
            idp = v['id']
            if idp in likes_map:
                ratings.append(likes_map[idp]['rating'])
                nbs.append(likes_map[idp]['NB'])
            else:
                ratings.append(None)
                nbs.append(None)

        g.vs['rating'] = ratings
        g.vs['NB'] = nbs

        # Add edges
        edge_tuples = []
        for _, row in df_edges_filtered.iterrows():
            src = idplace_to_index.get(row['source'])
            tgt = idplace_to_index.get(row['target'])
            if src is not None and tgt is not None:
                edge_tuples.append((src, tgt))
        g.add_edges(edge_tuples)
        ## Uncomment the next line if use only connected nodes
        g.delete_vertices([v.index for v in g.vs if v.degree() == 0])
        ## Delete self-loops edges
        g.delete_edges([e.index for e in g.es if e.source == e.target])
        g.simplify(multiple=True, loops=False)
        #print(g.vcount())

        graphs.append(g)

    return graphs

def parse_gspan_patterns(gspan_file_path):
    patterns = []
    current_labels = {}
    current_edges = []
    
    with open(gspan_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('t'):
                # Save previous pattern
                if current_labels or current_edges:
                    g = Graph()
                    g.add_vertices(len(current_labels))
                    g.vs['label'] = [current_labels[i] for i in range(len(current_labels))]
                    g.add_edges(current_edges)
                    g.es['label'] = [1] * len(current_edges)
                    patterns.append(g)
                    current_labels = {}
                    current_edges = []
            elif line.startswith('v'):
                _, vid, vlabel = line.split()
                current_labels[int(vid)] = int(vlabel)
            elif line.startswith('e'):
                _, src, dst, elabel = line.split()
                current_edges.append((int(src), int(dst)))
        # Save last pattern
        if current_labels or current_edges:
            g = Graph()
            g.add_vertices(len(current_labels))
            g.vs['label'] = [current_labels[i] for i in range(len(current_labels))]
            g.add_edges(current_edges)
            g.es['label'] = [1] * len(current_edges)
            patterns.append(g)
    return patterns

# Function for mapping the right graph IDs before the mining process

def write_gspan_input(graphs, output_filename):
    mapping = {}
    with open(output_filename, 'w') as f:
        for g_idx, g in enumerate(graphs):
            mapping[g_idx] = {}
            f.write(f"t # {g_idx}\n")
            
            # Asignación de nuevos números a los vértices y creación del mapeo.
            for new_id, vertex in enumerate(g.vs):
                #orig_id = vertex['name']  # Se asume que el atributo 'id' contiene el ID original.
                orig_id = vertex['id']  # Se asume que el atributo 'id' contiene el ID original.
                mapping[g_idx][new_id] = orig_id
                f.write(f"v {new_id} " + str(orig_id) + " \n")
            
            # Escritura de las aristas utilizando los nuevos números.
            for edge in g.es:
                src, tgt = edge.tuple
                f.write(f"e {src} {tgt} 1\n")
    return mapping

def find_max_subgraphs(lst_subgraphs):
    max_subgraph = []

    n = len(lst_subgraphs)

    for i in range(n):
        is_subgraph = False
        g1 = lst_subgraphs[i]
        
        for j in range(n):
            if i == j:
                continue

            g2 = lst_subgraphs[j]

            # If g1 has more or equal nodes than g2, we compare them
            if g1.vcount() <= g2.vcount():
                # Check if g1 is contained within g2 (subgraph)
                # Check that the nodes and edges match
                if set(g1.get_edgelist()).issubset(g2.get_edgelist()) and set(g1.vs["label"]).issubset(g2.vs["label"]):
                    is_subgraph = True
                    break

        if not is_subgraph:
            max_subgraph.append(g1)

    return max_subgraph

# Function to extract frequent subgraphs with gSpan

def subgraph_pattens_mining(minSup, input_filename, output_filename, max_size_patt):
    obj = alg.GSpan(input_filename, minSup, outputSingleVertices=False, maxNumberOfEdges=max_size_patt)
    #obj = alg.GSpan(input_filename, minSup, outputSingleVertices=False, maxNumberOfEdges=max_size_patt, 
    # outputGraphIds=True)
    obj.mine()
    frequentGraphs = obj.getFrequentSubgraphs()
    obj.save(output_filename)
    return frequentGraphs

def add_degree_and_betweenness(G):
    '''
    Calculates the degree and the  betweeness centrality of each node 
    and adds it as a 'degree' and 'bcentrality' attributes
    '''
    # 
    degree_values = G.degree(mode="in")
    G.vs['indegree'] = degree_values 

    betweenness_values = G.betweenness()
    G.vs['bcentrality'] = betweenness_values

    # Uncomment for normalizing the betweenness
    # betweenness_normalized = [b / ( (len(G.vs)-1)*(len(G.vs)-2) ) for b in betweenness_values]
    # G.vs['betweenness_normalized'] = betweenness_normalized

def processing_subgraph_ids_old(id_list, list_dataframes):
    rows = []
    ############print('ID_list -> ', id_list)
    for node_id in id_list:
        #print("el nodo :::: " , node_id)
        row = {'name': node_id}
        for i, df in enumerate(list_dataframes):
            #print(" buscar el nodo ", node_id )
            #print(" en ", df.head() )
            #print("1er name del df ----> ", df['name'][0])
            df_match = df[df['name'] == str(node_id)]
            #print(" df match :::::: " , df_match)
            if not df_match.empty:
                #print(" lo encuentrooooo df_match ----> ", df_match)
                #print (' ---> ', node_id)
                row[f'name_{i}'] = df_match.iloc[0].get('nom', None)
                #row[f'nbAvis_{i}'] = df_match.iloc[0].get('nbAvis', None)
                row[f'indegree_{i}'] = df_match.iloc[0].get('indegree', None)
                row[f'bcentrality_{i}'] = df_match.iloc[0].get('bcentrality', None)
                #row[f'huff_{i}'] = df_match.iloc[0].get('huff', None)
                row[f'mca_huff_{i}'] = df_match.iloc[0].get('mca_huff', None)
            else:   

                row[f'name_{i}'] = "None"
                #row[f'nbAvis_{i}'] = "None"
                row[f'indegree_{i}'] = "None"
                row[f'bcentrality_{i}'] = "None"
                #row[f'huff_{i}'] = "None"
                row[f'mca_huff_{i}'] = "None"

        #print("la row ", row)
        rows.append(row)

    #print(" procesar subgrafo ------ " , pd.DataFrame(rows).head())

    return pd.DataFrame(rows)

# Function for recovering and adding new metrics to the frequent vertices

def extract_vertices_patterns_ids_original(output_filename_original_id, list_dataframes):

    with open(output_filename_original_id, 'r') as f:
        lines = f.readlines()

    L_patterns = []
    current_nodes = []

    for line in lines:
        line = line.strip()
        if line.startswith('t #'):
            #print("current header -----> " , line)
            if current_nodes:
                L_patterns.append(processing_subgraph_ids(current_nodes, list_dataframes))
                current_nodes = []
        elif line.startswith('v'):
            #print("id node (1st loop) ----> ", line)
            parts = line.split()
            #print("parts :::: ", parts)
            if len(parts) >= 2: ### For getting the ID
            #if len(parts) >= 3: ### For getting the label
                #current_nodes.append(int(parts[1]))  # id original del nodo
                current_nodes.append(int(parts[2]))  # label original del nodo 
                #print("id node (1st loop) ----> ", int(parts[2]))

    if current_nodes:
        L_patterns.append(processing_subgraph_ids(current_nodes, list_dataframes))

    return L_patterns

def processing_subgraph_ids(id_list, list_dataframes):
    rows = []
    ############print('ID_list -> ', id_list)
    for node_id in id_list:
        #print("el nodo :::: " , node_id)
        row = {'id': node_id}
        for i, df in enumerate(list_dataframes):
            #print(" buscar el nodo ", node_id )
            #print(" en ", df.head() )
            #print("1er name del df ----> ", df['name'][0])
            df_match = df[df['id'] == int(node_id)]
            #print(" df match :::::: " , df_match)
            if not df_match.empty:
                #print(" lo encuentrooooo df_match ----> ", df_match)
                #print (' ---> ', node_id)
                row[f'name_{i}'] = df_match.iloc[0].get('nom', None)
                #row[f'nbAvis_{i}'] = df_match.iloc[0].get('nbAvis', None)
                row[f'rating_{i}'] = df_match.iloc[0].get('rating', None)
                row[f'NB_{i}'] = df_match.iloc[0].get('NB', None)
                row[f'indegree_{i}'] = df_match.iloc[0].get('indegree', None)
                row[f'bcentrality_{i}'] = df_match.iloc[0].get('bcentrality', None)
                #row[f'huff_{i}'] = df_match.iloc[0].get('huff', None)
                row[f'mca_huff_{i}'] = df_match.iloc[0].get('mca_huff', None)
            else:   

                row[f'name_{i}'] = "None"
                #row[f'nbAvis_{i}'] = "None"
                row[f'rating_{i}'] = "None"
                row[f'NB_{i}'] = "None"
                row[f'indegree_{i}'] = "None"
                row[f'bcentrality_{i}'] = "None"
                #row[f'huff_{i}'] = "None"
                row[f'mca_huff_{i}'] = "None"

        #print("la row ", row)
        rows.append(row)

    #print(" procesar subgrafo ------ " , pd.DataFrame(rows).head())

    return pd.DataFrame(rows)

# Function for recovering and adding new metrics to the frequent vertices

def extract_vertices_patterns_ids_original(output_filename_original_id, list_dataframes):

    with open(output_filename_original_id, 'r') as f:
        lines = f.readlines()

    L_patterns = []
    current_nodes = []

    for line in lines:
        line = line.strip()
        if line.startswith('t #'):
            #print("current header -----> " , line)
            if current_nodes:
                L_patterns.append(processing_subgraph_ids(current_nodes, list_dataframes))
                current_nodes = []
        elif line.startswith('v'):
            #print("id node (1st loop) ----> ", line)
            parts = line.split()
            #print("parts :::: ", parts)
            if len(parts) >= 2: ### For getting the ID
            #if len(parts) >= 3: ### For getting the label
                #current_nodes.append(int(parts[1]))  # id original del nodo
                current_nodes.append(int(parts[2]))  # label original del nodo 
                #print("id node (1st loop) ----> ", int(parts[2]))

    if current_nodes:
        L_patterns.append(processing_subgraph_ids(current_nodes, list_dataframes))

    return L_patterns

# Function for calculating the haversine distance (MCA Huff)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance between two (latitude, longitude) coordinates in kilometers.
    """
    R = 6371.0  # Earth radius in km

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Function for calculating the MCA Huff attractiveness

def compute_mca_huff(G, criteria_attrs, alpha=1, beta=1):
    """
    Compute the MCA Huff model for all nodes in a graph.
    
    Parameters:
    - G (igraph.Graph): The input graph with node attributes including lat/lon and criteria_attrs
    - criteria_attrs (list of str): List of node attributes used as attractiveness criteria
    - alpha (float): Exponent controlling the weight of attractiveness (default: 1)
    - beta (float): Exponent controlling the effect of distance (default: 1)

    Returns:
    - Updates G by adding a new vertex attribute 'mca_huff' for each node
    """

    # Step 1: Get coordinates and attributes for all nodes
    n = len(G.vs)
    latitudes = [v['latitude'] for v in G.vs]
    longitudes = [v['longitude'] for v in G.vs]

    # Step 2: Compute attractiveness scores for all nodes
    attractiveness = []
    for v in G.vs:
        score = 1.0
        for attr in criteria_attrs:
            val = v[attr]
            score *= float(val)  # multiplicative aggregation
        attractiveness.append(score ** alpha)

    # Step 3: Compute Huff probability for each node i over all j (excluding i)
    mca_huff = []
    for i in range(n):
        probs = []
        for j in range(n):
            if i == j:
                continue
            dist_ij = haversine_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
            if dist_ij == 0:
                continue  # Avoid division by zero
            score = attractiveness[j] / (dist_ij ** beta)
            probs.append(score)
        total = sum(probs)
        own_score = attractiveness[i]
        denom = own_score + total
        huff_value = own_score / denom if denom > 0 else 0
        mca_huff.append(huff_value)

    # Step 4: Add the MCA Huff values as a new node attribute
    G.vs['mca_huff'] = mca_huff

    return G

def creation_label_x_axis(edges, mode):
    #df_time = edges[['year', 'month']].drop_duplicates()
    df_time = edges[['year', 'month']].drop_duplicates()
    years = sorted(df_time['year'].unique())
    df_time['date'] = pd.to_datetime(df_time[['year', 'month']].assign(day=1))
    df_time = df_time.sort_values('date')
    #months = df_time['date'].dt.strftime('%Y-%m').tolist()
    months = df_time['date'].dt.strftime('%m-%Y').tolist()
    if mode == "month":
        return months
    else:
        return years

# Function for ploring metrics

def plot_metrics(df, metrics, labels, output_folder, mode, style=['science','ieee']):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    individual_figures = []
    subplot_labels = list(string.ascii_uppercase)

    for idx, metric in enumerate(metrics):
        columnas_mes = [f"{metric}_{i}" for i in range(len(labels))]
        df_tmp = df[['id'] + columnas_mes].set_index('id')
        names = df['name_0']

        font = {'family': 'serif', 'weight': 'bold', 'size': 12}
        matplotlib.rc('font', **font)

        df_tmp = df_tmp.fillna(0)

        title_x = "Year - month" if mode == "month" else "Year"

        title_dict = {
            "indegree": "In-degree centrality",
            "bcentrality": "Betweenness centrality",
            "mca_huff": "Huff model",
            "rating": "Rating (likes)",
            "NB": "Number of comments",
            "norm_nb": "Normalized number of comments",
            "norm_rating": "Normalized rating (likes)"
        }
        title_y = title_dict.get(metric, metric)

        # === Individual plot ===
        plt.style.use(style)
        fig, ax = plt.subplots(figsize=(10, 6))

        for place_id, fila in df_tmp.iterrows():
            ax.plot(labels, fila.values, label=f"{place_id}")

        ax.set_ylabel(title_y)
        ax.set_xlabel(title_x)
        ax.set_xticks(ticks=range(0, len(labels), 6))
        ax.set_xticklabels(labels[::6], rotation=90)
        ax.grid(True, axis='y', which='major')
        ax.legend(names, frameon=True, facecolor='white', edgecolor='black', fancybox=True)
        fig.tight_layout()

        # Save individual plot
        fig.savefig(os.path.join(output_folder, f"{metric}_evol.pdf"))
        individual_figures.append(fig)
        plt.close(fig)

    # === Grid PDF of all plots ===
    n_cols = 2
    n_rows = (len(metrics) + 1) // 2

    summary_pdf_path = os.path.join(output_folder, "summary_all_metrics.pdf")
    with PdfPages(summary_pdf_path) as pdf:
        fig_grid, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            columnas_mes = [f"{metric}_{i}" for i in range(len(labels))]
            df_tmp = df[['id'] + columnas_mes].set_index('id')
            names = df['name_0']
            df_tmp = df_tmp.fillna(0)
            ax = axes[idx]

            for place_id, fila in df_tmp.iterrows():
                ax.plot(labels, fila.values, label=f"{place_id}")

            title_y = title_dict.get(metric, metric)
            ax.set_ylabel(title_y)
            ax.set_xlabel(title_x)
            ax.set_xticks(ticks=range(0, len(labels), 6))
            ax.set_xticklabels(labels[::6], rotation=90)
            ax.legend(names, frameon=True, facecolor='white', edgecolor='black', fancybox=True)
            ax.grid(True, axis='y', which='major')
            ax.text(0.01, 0.98, subplot_labels[idx], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='left')

        # Hide unused subplots if metrics is odd
        for j in range(len(metrics), len(axes)):
            fig_grid.delaxes(axes[j])

        fig_grid.tight_layout()
        pdf.savefig(fig_grid)
        plt.close(fig_grid)

# Function for calculating the TF for nodes in patterns

from collections import Counter

def count_freq_vertices_patt(lst_nodes_patt, nodes_df):
    # Flatten the list of lists to count frequencies
    all_ids = [item for sublist in lst_nodes_patt for item in sublist]
    
    count = Counter(all_ids)
    #print(count)
    # Convert the counter to a DataFrame
    df_frecuencia = pd.DataFrame(count.items(), columns=['id', 'frequency'])
    
    # Join with the original dataframe to get the attributes
    df_result = pd.merge(df_frecuencia, nodes_df, on='id', how='left')
    
    # Reordenar columnas si es necesario
    columnas_deseadas = ['id', 'nom', 'latitude', 'longitude', 'frequency']
    df_result = df_result[columnas_deseadas]
    
    return df_result

# Function for ploting a heatmap from pattern' nodes frequency

def density_map_plotly_v0(df, output_folder, center_lat=None, center_lon=None):

    # Calculate the center if not exists
    if center_lat is None:
        center_lat = df['latitude'].mean()
    if center_lon is None:
        center_lon = df['longitude'].mean()
    
    fig = px.density_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        z="frequency",
        radius=20,
        center={"lat": center_lat, "lon": center_lon},
        zoom=12,
        mapbox_style="carto-positron",
        hover_name="nom",
        hover_data={"frequency": True, "latitude": False, "longitude": False},
        height=600
    )
    # Añadir scattermapbox para etiquetas permanentes
    scatter = go.Scattermapbox(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='markers+text',
        text=df['nom'],
        textposition='middle right',
        marker=dict(size=2, color='white', opacity=0.1),
        textfont=dict(size=14, color='black'),
        showlegend=False
    )
    
    ## Uncomment the following line for adding the name of places
    #fig.add_trace(scatter)
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.write_image(os.path.join(output_folder, "heatmap.pdf"), scale=1, width=1000, height=600)
    fig.show()

def summarize_graphs(lst_graphs, output_csv_path):
    num_vertices = [g.vcount() for g in lst_graphs]
    num_edges = [g.ecount() for g in lst_graphs]

    summary = {
        'max_vertices': max(num_vertices),
        'min_vertices': min(num_vertices),
        'mean_vertices': round(statistics.mean(num_vertices), 2),
        'max_edges': max(num_edges),
        'min_edges': min(num_edges),
        'mean_edges': round(statistics.mean(num_edges), 2),
    }

    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(output_csv_path, index=False)

    return df_summary


def parse_gspan_output(file_path):
    patterns = []
    current_pattern = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('t #'):
                # new pattern
                parts = line.split()
                current_pattern = {
                    'nb': int(parts[2]),
                    'vertices': [],
                    'support': int(parts[4])
                }
                patterns.append(current_pattern)
            elif line.startswith('v '):
                # adding enw vertice
                parts = line.split()
                current_pattern['vertices'].append(int(parts[2]))
    
    # Buolding a dataframe
    df = pd.DataFrame(patterns)
    df = df.sort_values('support', ascending=False).reset_index(drop=True)
    return df

def extend_vertices_patt(df_patrones, df_vertices):
    dfs = []
    for _, row in df_patrones.iterrows():
        df_v = df_vertices[df_vertices['id'].isin(row['vertices'])].copy()
        df_v['nb'] = row['nb']  # Add number of pattern
        df_v['support'] = row['support']  # Add number of pattern
        dfs.append(df_v)
    return pd.concat(dfs, ignore_index=True)


def frequence_by_vertice_patt(df_patts):

    # Calculating the frequency
    frequency = df_patts['id'].value_counts().reset_index()
    frequency.columns = ['id', 'frequency']
    
    # Calculiating unique features by vertice
    unique_features = df_patts.drop_duplicates('id')[['id', 'nom', 'latitude', 'longitude', 'typeR']]
    
    # CCombining freq + features
    df_final = pd.merge(unique_features, frequency, left_on='id', right_on='id')
    
    # Ordering the dataframe
    df_final = df_final.sort_values('frequency', ascending=False).reset_index(drop=True)
    
    return df_final[['id', 'nom', 'latitude', 'longitude', 'frequency']]


def density_map_plotly(df, output_folder, zoom_fig, city):

    if city == "Lille":
    # Calculate the center if not exists
        #center_lat = df['latitude'].mean()+0.015
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()+0.02
    else:
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean() 

    df['frequency_display'] = df['frequency'] + 1

    fig = px.density_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        z='frequency',
        radius=20,
        center={"lat": center_lat, "lon": center_lon},
        zoom=zoom_fig,
        mapbox_style="carto-positron",
        hover_name="nom",
        hover_data={"frequency": True, "latitude": False, "longitude": False},
        height=600,
        color_continuous_scale = 'viridis',
        custom_data = ['frequency']
    )

    df_freq1 = df[df['frequency'] == 1]

    # Add mark for less-frequency points
    fig.add_trace(go.Scattermapbox(
        lat=df_freq1['latitude'],
        lon=df_freq1['longitude'],
        mode='markers',
        marker=dict(
            size=6,
            color='rgb(68,1,84)',  # Color bajo de Viridis
            opacity=0.3
        ),
        text=df_freq1['nom'],
        hoverinfo='text',
        name='Frequency = 1'
    ))

    
    ## Uncomment the following line for adding the name of places
    #fig.add_trace(scatter)
    
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.write_image(output_folder, format='pdf', scale=1, height=500, width=1000)
    fig.show()

def filter_patts_by_types(df, patt_type):
    type_counts = df.groupby('nb')['typeR'].nunique()
    cluster_type_sets = df.groupby('nb')['typeR'].apply(set)
    target_clusters = cluster_type_sets[cluster_type_sets.apply(lambda s: patt_type.issubset(s))].index
    filtered_df = df[df['nb'].isin(target_clusters)]
    return filtered_df

def add_rating_graphs(lista_grafos, df_metricas):

    #if len(lista_grafos) != len(df_metricas.groupby(['year', 'month']).ngroups):
    #    raise ValueError("El número de grafos no coincide con los meses únicos en el DataFrame")
    
    for i, grafo in enumerate(lista_grafos):
        fecha_grafo = df_metricas[['year', 'month']].drop_duplicates().sort_values(['year', 'month']).iloc[i]
        year, month = fecha_grafo['year'], fecha_grafo['month']
        
        # Filtering metrics by year and month
        metricas_mes = df_metricas[(df_metricas['year'] == year) & (df_metricas['month'] == month)]
        
        # Create dics for quickly search
        ratings = dict(zip(metricas_mes['id'], metricas_mes['rating']))
        nbs = dict(zip(metricas_mes['id'], metricas_mes['NB']))
        
        # Add attributes to graphs
        for nodo in grafo.vs:
            id_nodo = nodo['id']
            nodo['rating'] = ratings.get(id_nodo, None)  # None if not exists
            nodo['NB'] = nbs.get(id_nodo, None)

def plot_likes_evolution(df, node_id, attributes, output_folder, style=['science','ieee']):

    size_font = 11

    node_data = df[df['id'] == node_id].copy()

    font = {'family': 'serif', 'weight': 'bold', 'size': size_font}
    matplotlib.rc('font', **font)
        
    # Create column with dates
    node_data['date'] = pd.to_datetime(
        node_data['year'].astype(str) + '-' + node_data['month'].astype(str) + '-01'
    )
    
    node_data.sort_values('date', inplace=True)
    #print(node_data)
    
    # Configure figure
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(12, 6))
       
    title_dict = {
        "indegree": "In-degree centrality",
        "bcentrality": "Betweenness centrality",
        "mca_huff": "Huff model",
        "rating": "Rating (like)",
        "NB": "Number of comments",
        "norm_nb": "Normalized number of comments",
        "norm_rating": "Normalized rating (likes)"
    }

    matplotlib.rcParams.update({'font.size': size_font})

    # Plot each feature
    lines = []
    for i, attr in enumerate(attributes):
        line = ax.plot(node_data['date'], node_data[attr], label=title_dict[attr])
        lines.append(line[0])
    

    #title_y = title_dict.get(attr, attributes)
    title_y = "Normalized Number of comments/Rating"
    title_x = "Month - year"
    legend = ax.legend(handles=lines,
                      title='Attributes',
                      frameon=True,
                      framealpha=0.9,
                      edgecolor='black',
                      facecolor='white',
                      loc='best')  # 'best' search the best position
    
    # Setting of the figure
    plt.subplots_adjust(right=0.8)    
    ax.set_ylabel(title_y, fontsize=size_font)
    ax.set_xlabel(title_x, fontsize=size_font)
    ax.xaxis.set_major_locator(MonthLocator(interval=3))  # Marcadores cada 3 meses
    ax.xaxis.set_major_formatter(DateFormatter('%m-%Y'))  # Formato MM-AAAA
    plt.xticks(rotation=90, fontsize=size_font)  # Rotación 90° 
    plt.yticks(fontsize=size_font) 
    ax.grid(True, axis='y', which='major')
    #ax.legend(names, frameon=True, facecolor='white', edgecolor='black', fancybox=True)
    fig.tight_layout()
    
    #fig.show()
    
    # Save in PDF format
    with PdfPages(os.path.join(output_folder, f"node_{node_id}_evolution.pdf")) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()


def normalize_ratings(df):
     # Ensure proper data types
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)

    # Sort by id, year, month
    df = df.sort_values(by=['id', 'year', 'month']).reset_index(drop=True)

    # Step 1 & 2: Calculate NB_acc and add_rating per id using custom loop
    def compute_group_norm(group):
        nb_acc = []
        add_rating = []
        current_nb_acc = 0
        current_add_rating = 0

        for _, row in group.iterrows():
            current_nb_acc += row['NB']
            current_add_rating += row['NB'] * row['rating']

            nb_acc.append(current_nb_acc)
            add_rating.append(current_add_rating)

        group['NB_acc'] = nb_acc
        group['add_rating'] = add_rating
        return group

    # Apply the logic to each id group
    df = df.groupby('id').apply(compute_group_norm).reset_index(drop=True)

    # Step 3: norm_rating = add_rating / NB_acc
    df['norm_rating'] = df['add_rating'] / df['NB_acc']

    # Step 4: aux = SUMIFS-style sum of NB by year and month (regardless of id)
    df['aux'] = df.groupby(['year', 'month'])['NB'].transform('sum')

    # Step 5: norm_nb = NB / aux
    df['norm_nb'] = df['NB'] / df['aux'] * 100

    return df

def plot_likes_evol_multi(df, node_ids, attributes, output_folder, style=['science','ieee']):
    
    # Configuration
    size_font = 10
    font = {'family': 'serif', 'weight': 'bold', 'size': size_font}
    matplotlib.rc('font', **font)
    plt.style.use(style)
    matplotlib.rcParams.update({'font.size': size_font})
    
    # Title dictionary
    title_dict = {
        "indegree": "In-degree centrality",
        "bcentrality": "Betweenness centrality",
        "mca_huff": "Huff model",
        "rating": "Rating (like)",
        "NB": "Number of comments",
        "norm_nb": "Normalized number of comments",
        "norm_rating": "Normalized rating (likes)"
    }
    
    if len(node_ids) == 4:
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)
        axs = axs.ravel()
        
        # Create a list to store all line handles for the legend
        all_lines = []
        
        # Plot each node
        for i, node_id in enumerate(node_ids):
            # Get and prepare node data
            node_data = df[df['id'] == node_id].copy()
            node_data['date'] = pd.to_datetime(
                node_data['year'].astype(str) + '-' + node_data['month'].astype(str) + '-01'
            )
            node_data.sort_values('date', inplace=True)
            
            # Plot attributes
            lines = []
            for attr in attributes:
                line = axs[i].plot(node_data['date'], node_data[attr], label=title_dict[attr])
                lines.append(line[0])
            
            # Store lines for legend (only once per attribute)
            if i == 0:
                all_lines = lines.copy()
            
            # Add subplot label (A), B), etc.)
            axs[i].text(0.02, 0.98, f'{chr(65+i)}', 
                    transform=axs[i].transAxes,
                    fontsize=12,
                    fontweight='bold',
                    verticalalignment='top')
            
            # Configure subplot
            #axs[i].set_title(f'Node {node_id}', pad=10)
            if (i == 0) or (i == 2):
                axs[i].set_ylabel("Normalized nb of comments / rating", fontsize=size_font)
            else:
                axs[i].set_ylabel("")

            if (i == 2) or (i == 3):
                axs[i].set_xlabel("Month - year", fontsize=size_font)
            else:
                axs[i].set_xlabel("")
            axs[i].xaxis.set_major_locator(MonthLocator(interval=3))
            axs[i].xaxis.set_major_formatter(DateFormatter('%m-%Y'))
            axs[i].tick_params(axis='x', rotation=90)
            axs[i].grid(True, axis='y')
        
        # Create a single shared legend at the bottom
        fig.legend(handles=all_lines,
                #title='Attributes',
                frameon=True,
                framealpha=0.9,
                edgecolor='black',
                facecolor='white',
                loc='lower center',
                bbox_to_anchor=(0.5, -0.05),
                ncol=len(attributes))
        
        # Adjust layout to make room for the legend
        plt.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.1)
        
        # Save to PDF
        os.makedirs(output_folder, exist_ok=True)
        with PdfPages(os.path.join(output_folder, "multi_node_evolution.pdf")) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    else:
        # Create figure with 1x2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.7))
        gs = fig.add_gridspec(1, 2, hspace=0.25, wspace=0.2)
        axs = axs.ravel()
        
        # Create a list to store all line handles for the legend
        all_lines = []
        
        # Plot each node
        for i, node_id in enumerate(node_ids):
            # Get and prepare node data
            node_data = df[df['id'] == node_id].copy()
            node_data['date'] = pd.to_datetime(
                node_data['year'].astype(str) + '-' + node_data['month'].astype(str) + '-01'
            )
            node_data.sort_values('date', inplace=True)
            
            # Plot attributes
            lines = []
            for attr in attributes:
                line = axs[i].plot(node_data['date'], node_data[attr], label=title_dict[attr])
                lines.append(line[0])
            
            # Store lines for legend (only once per attribute)
            if i == 0:
                all_lines = lines.copy()
            
            # Add subplot label (A), B), etc.)
            axs[i].text(0.02, 0.98, f'{chr(65+i)}', 
                    transform=axs[i].transAxes,
                    fontsize=12,
                    fontweight='bold',
                    verticalalignment='top')
            
            # Configure subplot
            #axs[i].set_title(f'Node {node_id}', pad=10)
            if (i == 0) or (i == 2):
                axs[i].set_ylabel("Normalized nb of comments / rating", fontsize=size_font)
            else:
                axs[i].set_ylabel("")

            if (i == 2) or (i == 3):
                axs[i].set_xlabel("Month - year", fontsize=size_font)
            else:
                axs[i].set_xlabel("")
            axs[i].xaxis.set_major_locator(MonthLocator(interval=3))
            axs[i].xaxis.set_major_formatter(DateFormatter('%m-%Y'))
            axs[i].tick_params(axis='x', rotation=90)
            axs[i].grid(True, axis='y')
        
        # Create a single shared legend at the bottom
        fig.legend(handles=all_lines,
                #title='Attributes',
                frameon=True,
                framealpha=0.9,
                edgecolor='black',
                facecolor='white',
                loc='lower center',
                bbox_to_anchor=(0.5, -0.05),
                ncol=len(attributes))
        
        # Adjust layout to make room for the legend
        plt.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.1)
        
        # Save to PDF
        os.makedirs(output_folder, exist_ok=True)
        with PdfPages(os.path.join(output_folder, "multi_node_evolution.pdf")) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()


def extract_mca_huff_series(graph_list, target_node_id, output_folder, window_norm = 3):
    data = []
    
    for month_idx, graph in enumerate(graph_list, start=1):
        try:
            node = graph.vs.find(id = int(target_node_id))
            mca_huff = node['mca_huff'] if 'mca_huff' in node.attributes() else 0
        except:
            mca_huff = 0
        
        data.append({
            'id': int(target_node_id),
            'mca_huff': float(mca_huff)  # Aseguramos tipo float
        })
    
    df_huff_serie = pd.DataFrame(data)
    df_huff_serie['norm_huff'] = df_huff_serie['mca_huff'].rolling(window=window_norm, center=True).mean()

    ## Prediction: lineal

    X = np.arange(len(df_huff_serie)).reshape(-1, 1)
    y = df_huff_serie['mca_huff'].values
    model_li = LinearRegression().fit(X, y)
    df_huff_serie['line_regre'] = model_li.predict(X)

    ## Prediction: polynomial

    degree = 3
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model_po = LinearRegression().fit(X_poly, y)
    df_huff_serie['poli_regre'] = model_po.predict(X_poly)

    df_huff_serie.fillna(0)

    #print(np.any(np.isnan(X)))       # Should be False
    #print(np.any(np.isinf(X)))       # Should be False
    #print(np.max(np.abs(X)))         # Should be within a reasonable range
    
    ## Prediction: logaritmic
    X = np.arange(1, len(y) + 1).reshape(-1, 1)
    X_log = np.log(X)
    y = df_huff_serie['mca_huff'].values
    model_lo = LinearRegression().fit(X_log, y)
    df_huff_serie['log_regre'] = model_lo.predict(X_log)
    #X = np.arange(len(df_huff_serie)).reshape(-1, 1)
    #y = df_huff_serie['mca_huff'].values
    #model_lo = LogisticRegression().fit(X, y)
    #df_huff_serie['log_regre'] = model_lo.predict(X)

    coefs = []
    coef_li = model_li.coef_
    #coefs.append(coef_li)
    #print(coef_li)
    inter_li = model_li.intercept_
    #coefs.append(inter_li)
    #print(inter_li)
    coef_po = model_po.coef_
    #coefs.append(coef_po)
    #print(coef_po)
    inter_po = model_po.intercept_
    #coefs.append(inter_po)
    #print(inter_po)
    coef_lo = model_lo.coef_
    coefs.append(coef_lo)
    #print(coef_lo)
    inter_lo = model_lo.intercept_
    coefs.append(inter_lo)
    #print(inter_lo)
    
    #print(coefs)

    #file_out = os.path.join(output_folder, f"{target_node_id}_coefs.txt")
    #with open(file_out, "w") as file:
    #    for item in coefs:
    #        file.write(f"{item}\n")  # Each item on a new line

    return df_huff_serie


def plot_mca_huff_trend(df_huff_serie, node_id, attributes, month_labels, output_folder, style = ['science', 'ieee']):
    
    size_font = 11

    # Apply scientific plotting style
    font = {'family': 'serif', 'weight': 'bold', 'size': size_font}
    matplotlib.rc('font', **font)
    plt.style.use(style)
    
    # Initialize figure with specific dimensions
    fig, ax = plt.subplots(figsize=(12, 6))

    title_y = "Huff attractiveness"
    title_x = "Month - year"

    # Create line plot with markers
    ax.plot(month_labels, df_huff_serie[attributes[0]], color= "red", linestyle='--')
    ax.plot(month_labels, df_huff_serie[attributes[1]], color= "black", linestyle='-')
    matplotlib.rcParams.update({'font.size': size_font})
    
    # Formatting
    ax.set_ylabel(title_y, fontsize=size_font)
    ax.set_xlabel(title_x, fontsize=size_font)
    ax.set_xticks(month_labels[::3])  # Every 3rd month
    ax.set_xticklabels(month_labels[::3])
    plt.xticks(rotation=90, fontsize=size_font)  # Rotación 90° 
    plt.yticks(fontsize=size_font)
    ax.grid(True, axis='y', which='major')
    fig.tight_layout()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=90, ha='right')
    
    # Adjust layout to prevent label clipping
    fig.tight_layout()
    #plt.show()

    # Save in PDF format
    with PdfPages(os.path.join(output_folder, f"node_{node_id}_huff.pdf")) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def plot_mca_huff_trend_multi(graph_list, node_ids, attributes, month_labels, output_folder, style=['science', 'ieee']):
    
    # Configuration
    size_font = 10
    font = {'family': 'serif', 'weight': 'bold', 'size': size_font}
    matplotlib.rc('font', **font)
    plt.style.use(style)
    matplotlib.rcParams.update({'font.size': size_font})
    
    if len(node_ids) == 4:
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.2)
        axs = axs.ravel()

        #print(month_labels)
        
        # Plot each node
        for i, node_id in enumerate(node_ids):

            df_huff_serie = extract_mca_huff_series(graph_list, node_id, output_folder)
            df_huff_serie['huff_trend'] = df_huff_serie['mca_huff'].rolling(window=3, center=True).mean()

            ax = axs[i]
            values = []
        
            # Plot with consistent styling
            line = ax.plot(month_labels, df_huff_serie[attributes[0]], color= "red", linestyle='--')
            line = ax.plot(month_labels, df_huff_serie[attributes[1]], color= "black", linestyle='-')
            
            # Subplot letter label
            axs[i].text(0.02, 0.98, f'{chr(65+i)}', 
                    transform=axs[i].transAxes,
                    fontsize=12,
                    fontweight='bold',
                    verticalalignment='top')
            
            # Configure axes
            if (i == 0) or (i == 2):
                axs[i].set_ylabel("Huff attractiveness", fontsize=size_font)
            else:
                axs[i].set_ylabel("")

            if (i == 2) or (i == 3):
                axs[i].set_xlabel("Month - year", fontsize=size_font)
            else:
                axs[i].set_xlabel("")

            axs[i].set_xticks(month_labels[::3])
            axs[i].set_xticklabels(month_labels[::3])
            #ax.tick_params(axis='x', rotation=90)
            #ax.grid(True, axis='y', linestyle=':', alpha=0.6)
            axs[i].tick_params(axis='x', rotation=90)
            axs[i].grid(True, axis='y')
        
        # Save combined plot
        with PdfPages(os.path.join(output_folder, "multi_node_huff.pdf")) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    else:
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
        gs = fig.add_gridspec(1, 2, hspace=0.25, wspace=0.2)
        axs = axs.ravel()

        #print(month_labels)
        
        # Plot each node
        for i, node_id in enumerate(node_ids):

            df_huff_serie = extract_mca_huff_series(graph_list, node_id, output_folder)
            df_huff_serie['huff_trend'] = df_huff_serie['mca_huff'].rolling(window=3, center=True).mean()

            ax = axs[i]
            values = []
        
            # Plot with consistent styling
            line = ax.plot(month_labels, df_huff_serie[attributes[0]], color= "red", linestyle='--')
            line = ax.plot(month_labels, df_huff_serie[attributes[1]], color= "black", linestyle='-')
            
            # Subplot letter label
            axs[i].text(0.02, 0.98, f'{chr(65+i)}', 
                    transform=axs[i].transAxes,
                    fontsize=12,
                    fontweight='bold',
                    verticalalignment='top')
            
            # Configure axes
            if (i == 0) or (i == 2):
                axs[i].set_ylabel("Huff attractiveness", fontsize=size_font)
            else:
                axs[i].set_ylabel("")

            if (i == 2) or (i == 3):
                axs[i].set_xlabel("Month - year", fontsize=size_font)
            else:
                axs[i].set_xlabel("")

            axs[i].set_xticks(month_labels[::3])
            axs[i].set_xticklabels(month_labels[::3])
            #ax.tick_params(axis='x', rotation=90)
            #ax.grid(True, axis='y', linestyle=':', alpha=0.6)
            axs[i].tick_params(axis='x', rotation=90)
            axs[i].grid(True, axis='y')
        
        # Save combined plot
        with PdfPages(os.path.join(output_folder, "multi_node_huff.pdf")) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def compute_huff_regression(graph_list, target_node_id, output_folder, window_norm = 3):
    data = []
    
    for month_idx, graph in enumerate(graph_list, start=1):
        try:
            node = graph.vs.find(id = int(target_node_id))
            mca_huff = node['mca_huff'] if 'mca_huff' in node.attributes() else 0
        except:
            mca_huff = 0
        
        data.append({
            'id': int(target_node_id),
            'mca_huff': float(mca_huff)  # Aseguramos tipo float
        })
    
    df_huff_serie = pd.DataFrame(data)
    df_huff_serie['norm_huff'] = df_huff_serie['mca_huff'].rolling(window=window_norm, center=True).mean()
    
    ## Prediction: logaritmic
    y = df_huff_serie['mca_huff'].values
    X = np.arange(1, len(y) + 1).reshape(-1, 1)
    X_log = np.log(X)
    model_lo = LinearRegression().fit(X_log, y)
    df_huff_serie['log_regre'] = model_lo.predict(X_log)
    #X = np.arange(len(df_huff_serie)).reshape(-1, 1)
    #y = df_huff_serie['mca_huff'].values
    #model_lo = LogisticRegression().fit(X, y)
    #df_huff_serie['log_regre'] = model_lo.predict(X)

    coefs = []
    coef_lo = model_lo.coef_
    coefs.append(coef_lo)
    #print(coef_lo)
    inter_lo = model_lo.intercept_
    coefs.append(inter_lo)
    #print(inter_lo)
    
    #print(coefs)

    #print("node --> ", node['id'])
    #print("coef --> ",coef_lo[0])
    #print("inter --> ",inter_lo)

    aux = []
    aux.append(node['id'])
    aux.append(node['nom'])
    aux.append(coef_lo[0])
    aux.append(inter_lo)

    #file_out = os.path.join(output_folder, f"{target_node_id}_coefs.txt")
    #with open(file_out, "w") as file:
    #    for item in coefs:
    #        file.write(f"{item}\n")  # Each item on a new line

    return aux


def haversine(lat1, lon1, lat2, lon2):
    ## Calculate the Haversine distance in kilometers
    R = 6371.0  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * asin(sqrt(a))

# Function for calculating the Huff attractiveness

def attract_huff(g: Graph, alpha=1.0, beta=1.0, attr='indegree'):

    alpha = float(alpha)
    beta = float(beta)
    
    n = len(g.vs)
    mca_scores = [0.0] * n

    for i, v_i in enumerate(g.vs):
        lat_i = v_i['latitude']
        lon_i = v_i['longitude']

        numerators = []
        denominator = 0.0

        for j, v_j in enumerate(g.vs):
            if i == j:
                numerators.append(0.0)
                continue

            lat_j = v_j['latitude']
            lon_j = v_j['longitude']

            try:
                A_j_raw = float(v_j[attr])
            except (ValueError, TypeError):
                A_j_raw = 1e-6

            A_j = (A_j_raw if A_j_raw > 0 else 1e-6) ** alpha
            D_ij = haversine(lat_i, lon_i, lat_j, lon_j) ** beta

            if D_ij == 0:
                numerators.append(0.0)
                continue

            value = A_j / D_ij
            numerators.append(value)
            denominator += value

        probabilities = [val / denominator if denominator > 0 else 0.0 for val in numerators]

        for j in range(n):
            mca_scores[j] += probabilities[j]

        total_score = sum(mca_scores)
        normalized_scores = [x / total_score if total_score > 0 else 0 for x in mca_scores]

    g.vs['mca_huff'] = normalized_scores
    #g.vs['mca_huff'] = mca_scores
    return g