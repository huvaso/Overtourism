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

from math import radians, sin, cos, sqrt, atan2
import plotly.graph_objects as go

from PAMI.extras.stats import graphDatabase as alg
from PAMI.subgraphMining.basic import gspan as alg
from PAMI.extras.visualize import graphs as vis

import overpy
from shapely.geometry import Polygon
import folium
from folium import plugins
import shapely.geometry

### Other libraries
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import plotly.express as px
import h3

from functools import reduce
from typing import List, Dict, Tuple

########################################

# Function for recovering CSV files for building SEASONS graph
# input: paths to edges and vertices files
# output: two dataframes containing vertices and edges

def read_files_season(path_edges, path_vertices):
    '''
    Function for recovering CSV files for building SEASONS graph
    Parameters:
    - paths to edges and vertices files

    Returns:
    - two dataframes containing vertices and edges
    '''
    vertices = pd.read_csv(path_vertices,sep="\t")
    edges = pd.read_csv(path_edges,sep="\t") 
    edges = edges.rename(columns={"gid_from":"source","gid_to":"target"})
    return vertices, edges


# Function for visualizing an igraph

def graph_visu(gs):
    """
    Plot a graph using matplotlib
    """
    layout = gs.layout("fr")  # "fr" = Fruchterman-Reingold layout
    fig, ax = plt.subplots(figsize=(8, 8))
    ig.plot(
        gs, 
        target=ax, 
        layout=layout, 
        vertex_size=30, 
        vertex_color="lightblue", 
        edge_width=1, 
        edge_color="gray",  # Color for edges
        vertex_label=[str(v.index) for v in gs.vs],  # Show verrices' names
        vertex_label_size=10
    )
    plt.show()

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

# Function for calculating MCA Huff model

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

# Function for mapping the nodes' IDs

def mapping_graphs(lst_graphs):
    """
    Function for mapping the nodes' IDs
    Parameters:
    - graphs (list): List of igraph.Graph objects, one for each time (e.g., months)

    Returns:
    - dic: A dictionary {graph_idx: {internal_id: original_id}}
    """
    id_mapping = {}  # Dictionary to store the mappings {graph_idx: {internal_id: original_id}}
    
    for i, graph in enumerate(lst_graphs):
        id_mapping[i] = {v.index: v["id"] for v in graph.vs} 
    
    return id_mapping

# Function for creating a tct file to store graphs for gSpan

def create_graph_gspan(lst_graphs, output_filename):
    '''
    Function for mapping a graph into a ,TXT file to be used by gSpan
    Parameters:
    - graphs (list): List of igraph.Graph objects, one for each time (e.g., months)
    - output_filename (str): Path to the output file.

    Returns:
    - dic: A dictionary {graph_idx: {internal_id: original_id}}
    - file: A .TXT file called 'graphs_gspam.txt'
    '''
    dic_vertex = []
    with open(output_filename, 'w') as f:
        for i in range(len(lst_graphs)):
            f.write("t # " + str(i) + '\n')
            for v in lst_graphs[i].vs:
                f.write(f"v {v['id']}" + ' 1 \n')
                a = (i, v['id'])
            for e in lst_graphs[i].es:
                source_vertex = lst_graphs[i].vs[e.source]['id']
                target_vertex = lst_graphs[i].vs[e.target]['id']
                f.write(f"e {source_vertex} {target_vertex}" + ' 101 \n')
            dic_vertex.append(a)
    return dic_vertex

# Function for mapping the right graph IDs before the mining process

def write_gspan_input(graphs, output_filename):
    """
    Writes the input file for gSpan from a list of igraph graphs.
    Generates a mapping per graph that relates the assigned consecutive number to the original ID.

    Parameters:
    - graphs (list): List of igraph.Graph objects, one for each time (e.g., months)
    - output_filename (str): Path to the output file.

    Returns:
    dict: Dictionary with the mapping for each graph, for example:
    { 0: {0: 7174290, 1: 12208448, 2: 3618712}, 1: {0: 3618712, 1: 7174290, 2: 9938243} }
    """
    mapping = {}
    with open(output_filename, 'w') as f:
        for g_idx, g in enumerate(graphs):
            mapping[g_idx] = {}
            f.write(f"t # {g_idx}\n")
            
            # Asignación de nuevos números a los vértices y creación del mapeo.
            for new_id, vertex in enumerate(g.vs):
                orig_id = vertex['id']  # Se asume que el atributo 'id' contiene el ID original.
                mapping[g_idx][new_id] = orig_id
                f.write(f"v {new_id} " + str(orig_id) + " \n")
            
            # Escritura de las aristas utilizando los nuevos números.
            for edge in g.es:
                src, tgt = edge.tuple
                f.write(f"e {src} {tgt} 1\n")
    return mapping

# Function to extract frequent subgraphs with gSpan

def subgraph_pattens_mining(minSup, input_filename, output_filename, max_size_patt):
    """
    Execute the gSpan algorith on a .TXT file containing the igraph.Graph objects

    Parameters:
    - input_filename (str): Path to the input file
    - output_filename (str): Path to the output file
    - max_size_patt (int): Max number of edges

    Returns:
    - obj: A list of frequent subgraphs
    - file : The list of fequent subgraphs 
    """
    obj = alg.GSpan(input_filename, minSup, outputSingleVertices=False, maxNumberOfEdges=max_size_patt)
    #obj = alg.GSpan(input_filename, minSup, outputSingleVertices=False, maxNumberOfEdges=max_size_patt, outputGraphIds=True)
    obj.mine()
    frequentGraphs = obj.getFrequentSubgraphs()
    obj.save(output_filename)
    return frequentGraphs

# Funtion for mapping the right ID vertices from the original graph

def recover_original_ids_in_file(mapping_for_graph, input_filename, output_filename):
    """
    Replaces the vertex numbers in a frequent pattern file (gSpan output) with the original IDs, using the mapping corresponding to a graph.

    Parameters:
    mapping_for_graph (dict): Dictionary mapping the new numbers to the original IDs.
    input_filename (str): Input file (gSpan patterns).
    output_filename (str): Output file with the original IDs.
    """
    with open(input_filename, 'r') as fin, open(output_filename, 'w') as fout:
        for line in fin:
            parts = line.strip().split()
            if not parts:
                continue
            # Si es línea de vértice: "v <nuevo_id> <label>"
            if parts[0] == 'v':
                try:
                    new_id = int(parts[1])
                except ValueError:
                    fout.write(line)
                    continue
                label = parts[2] if len(parts) > 2 else ""
                original_id = mapping_for_graph.get(new_id, new_id)
                fout.write(f"v {original_id} {label}\n")
            # Si es línea de arista: "e <nuevo_id1> <nuevo_id2> <label>"
            elif parts[0] == 'e':
                try:
                    new_id1 = int(parts[1])
                    new_id2 = int(parts[2])
                except ValueError:
                    fout.write(line)
                    continue
                label = parts[3] if len(parts) > 3 else ""
                original_id1 = mapping_for_graph.get(new_id1, new_id1)
                original_id2 = mapping_for_graph.get(new_id2, new_id2)
                fout.write(f"e {original_id1} {original_id2} {label}\n")
            else:
                # Para líneas que no sean de vértices o aristas (por ejemplo, las cabeceras 't')
                fout.write(line)

# Function for calculating the in_degree and the betweness centrality in a graph

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

# Function for recovering and adding new metrics to the frequent vertices

def processing_subgraph_ids(id_list, list_dataframes):
    rows = []
    ############print('ID_list -> ', id_list)
    for node_id in id_list:
        row = {'id': node_id}
        for i, df in enumerate(list_dataframes):
            #print(" buscar el nodo ", node_id )
            #print(" en ", df.head() )
            df_match = df[df['id'] == node_id]
            #print(" df match :::::: " , df_match)
            if not df_match.empty:
                #print (' ---> ', node_id)
                row[f'name_{i}'] = df_match.iloc[0].get('name', None)
                row[f'nbAvis_{i}'] = df_match.iloc[0].get('nbAvis', None)
                row[f'indegree_{i}'] = df_match.iloc[0].get('indegree', None)
                row[f'bcentrality_{i}'] = df_match.iloc[0].get('bcentrality', None)
                #row[f'huff_{i}'] = df_match.iloc[0].get('huff', None)
                row[f'mca_huff_{i}'] = df_match.iloc[0].get('mca_huff', None)
            else:
                #row[f'name_{i}'] = "None"
                row[f'nbAvis_{i}'] = "None"
                row[f'indegree_{i}'] = "None"
                row[f'bcentrality_{i}'] = "None"
                #row[f'huff_{i}'] = "None"
                row[f'mca_huff_{i}'] = "None"
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
            #print(" ----- T " , line)
            #print(current_nodes)
            if current_nodes:
                L_patterns.append(processing_subgraph_ids(current_nodes, list_dataframes))
                current_nodes = []
        elif line.startswith('v'):
            #print(" ----- V ", line)
            parts = line.split()
            #print("parts :::: ", parts)
            if len(parts) >= 2: ### For getting the ID
            #if len(parts) >= 3: ### For getting the label
                #current_nodes.append(int(parts[1]))  # id original del nodo
                current_nodes.append(int(parts[2]))  # label original del nodo 
    if current_nodes:
        L_patterns.append(processing_subgraph_ids(current_nodes, list_dataframes))

    return L_patterns

# Function for building igraph objects from .TXT file generated by gSpan (with labels)

def parse_gspan_patterns(gspan_file_path):
    """
    Parse gSpan output file into a list of igraph.Graph pattern objects.
    """
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

# Function for building igraph objects from .TXT file generated by gSpan (with labels)

def parse_gspan_patterns_h3(gspan_file_path):
    """
    Parse gSpan output file into a list of igraph.Graph pattern objects.
    """
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
                current_labels[str(vid)] = str(vlabel)
            elif line.startswith('e'):
                _, src, dst, elabel = line.split()
                current_edges.append((str(src), str(dst)))
        # Save last pattern
        if current_labels or current_edges:
            g = Graph()
            g.add_vertices(len(current_labels))
            g.vs['label'] = [current_labels[i] for i in range(len(current_labels))]
            g.add_edges(current_edges)
            g.es['label'] = [1] * len(current_edges)
            patterns.append(g)
    return patterns

# Function for ploring metrics

def plot_metrics(df, metrics, output_folder="metrics_ot"):
    """
    Plot metrics by pattern and by time (months).

    Args:
    df DataFrame: dataframe with metrics by nodes
    metrics list: a list of metrics should be ploted
    directory: the output directory storing all figures

    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for metric in metrics:
        columnas_mes = [f"{metric}_{i}" for i in range(len(metrics)+1)]
        df_tmp = df[['id'] + columnas_mes].set_index('id')
        names = df['name_0']

        df_tmp.fillna(0)
        #print(df_tmp)
        
        plt.figure(figsize=(10, 6))
        for place_id, fila in df_tmp.iterrows():
            plt.plot(range(1,6), fila.values, label=f"{place_id}")
        
        #plt.title(f"Evolución temporal de {metrica}")
        plt.xlabel("Months")
        plt.ylabel(metric)
        plt.xticks(range(1, 6), [f"Month {i}" for i in range(1, 6)])
        #plt.grid(True)
        plt.tight_layout()
        plt.legend(names)
        plt.savefig(os.path.join(output_folder, f"{metric}_evolucion.png"))
        plt.close()

# Function for extracting maximal subgraphs

def find_max_subgraphs(lst_subgraphs):
    """
    Filters and returns the maximal subgraphs from a list of frequent subgraphs.

    Args:
    list_subgraphs (list of igraph.Graph): List of frequent subgraphs.

    Returns:
    list of igraph.Graph: Maximal subgraphs (not subgraphs of any other graphs).
    """
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

# Function for calculating the TF for nodes in patterns

def count_freq_vertices_patt(lst_nodes_patt, nodes_df):
    # Flatten the list of lists to count frequencies
    all_ids = [item for sublist in lst_nodes_patt for item in sublist]
    
    count = Counter(all_ids)
    
    # Convert the counter to a DataFrame
    df_frecuencia = pd.DataFrame(count.items(), columns=['id', 'frequency'])
    
    # Join with the original dataframe to get the attributes
    df_result = pd.merge(df_frecuencia, nodes_df, on='id', how='left')
    
    # Reordenar columnas si es necesario
    columnas_deseadas = ['id', 'name', 'latitude', 'longitude', 'frequency']
    df_result = df_result[columnas_deseadas]
    
    return df_result

# Function for ploting a heatmap from pattern' nodes frequency

def density_map_plotly(df, center_lat=None, center_lon=None):

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
        hover_name="name",
        hover_data={"frequency": True, "latitude": False, "longitude": False},
        height=600
    )

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()



def get_h3(lat, lon, resolution):
    return h3.geo_to_h3(lat, lon, resolution)

def adding_h3_by_node(df, resolution, lat_col='latitude', lon_col='longitude', id_col='id'):
    df = df.copy()
    # Calculamos la celda H3 para cada nodo y la guardamos en una nueva columna.
    df['h3_id'] = df.apply(lambda row: get_h3(row[lat_col], row[lon_col], resolution), axis=1)

    return df

def group_mean_mode(df, id_col='h3_id'):
    # Lista para almacenar los resultados de cada columna
    funciones = {}

    for col in df.columns:
        if col == id_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            funciones[col] = 'mean'
        else:
            funciones[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan

    # Agrupar usando las funciones detectadas
    df_agrupado = df.groupby(id_col).agg(funciones).reset_index()
    return df_agrupado

def create_edges_from_h3(df_aristas, df_nodos, resolution=7):
    
    df_nodos = df_nodos.copy()
    df_nodos['h3_id'] = df_nodos.apply(lambda row: get_h3(row['latitude'], row['longitude'], resolution), axis=1)
    
    # Create a dictionary that maps each node id to its H3 cell
    mapeo_nodos = df_nodos.set_index('id')['h3_id'].to_dict()
    
    # Assign to each edge the H3 cell of its origin and destination node
    df_aristas = df_aristas.copy()
    df_aristas['h3_origin'] = df_aristas['source'].map(mapeo_nodos)
    df_aristas['h3_destino'] = df_aristas['target'].map(mapeo_nodos)
    
    # Group the edges by the source and destination H3 cells to obtain the weights
    df_aristas_agregadas = (df_aristas.groupby(['h3_origin', 'h3_destino']).size().reset_index(name='weight'))
    
    return df_aristas_agregadas


def build_graph_h3(df_agregadas, directed=True):
    # Extract the list of unique cells (nodes) present in source and destination
    celdas_h3 = set(df_agregadas['h3_origin']).union(set(df_agregadas['h3_destino']))
    celdas_h3 = list(celdas_h3)
    # Create a mapping from each H3 cell to a consecutive index (needed for igraph)
    mapeo = {celda: idx for idx, celda in enumerate(celdas_h3)}
    
    # Create list of edges and weight accumulator.
    lista_aristas = []
    lista_pesos = []
    
    # For each row in the DataFrame, convert the source and destination cells to indexes
    for _, row in df_agregadas.iterrows():
        origen = row['h3_origin']
        destino = row['h3_destino']
        peso = row['weight']
        lista_aristas.append((mapeo[origen], mapeo[destino]))
        lista_pesos.append(peso)
        
    # Graph building
    g = ig.Graph(directed=directed)
    g.add_vertices(len(celdas_h3))
    
    # Assign the original H3 identifier to each vertex so that it can be identified
    g.vs["h3_id"] = celdas_h3

    # Add edges and weights
    g.add_edges(lista_aristas)
    g.es["weight"] = lista_pesos
    
    return g

# Function for mapping the right graph IDs before the mining process

def write_gspan_input_h3(graphs, output_filename):
    """
    Writes the input file for gSpan from a list of igraph graphs.
    Generates a mapping per graph that relates the assigned consecutive number to the original ID.

    Parameters:
    - graphs (list): List of igraph.Graph objects, one for each time (e.g., months)
    - output_filename (str): Path to the output file.

    Returns:
    dict: Dictionary with the mapping for each graph, for example:
    { 0: {0: 7174290, 1: 12208448, 2: 3618712}, 1: {0: 3618712, 1: 7174290, 2: 9938243} }
    """
    mapping = {}
    with open(output_filename, 'w') as f:
        for g_idx, g in enumerate(graphs):
            mapping[g_idx] = {}
            f.write(f"t # {g_idx}\n")
            
            # Asignación de nuevos números a los vértices y creación del mapeo.
            for new_id, vertex in enumerate(g.vs):
                orig_id = vertex['h3']  # Se asume que el atributo 'id' contiene el ID original.
                mapping[g_idx][new_id] = orig_id
                f.write(f"v {new_id} " + str(orig_id) + " \n")
            
            # Escritura de las aristas utilizando los nuevos números.
            for edge in g.es:
                src, tgt = edge.tuple
                f.write(f"e {src} {tgt} 1\n")
    return mapping

# Function for buidling new nodes and edges grouping them in H3s

def group_nodes_by_h3(lst_graphs, resolution):
    grouped_graphs = []

    for g in lst_graphs:
        # Step 1: Extract coordinates and attributes from each node
        coords = [(v['latitude'], v['longitude']) for v in g.vs]
        nbAvis = g.vs['nbAvis']
        types = g.vs['typeR']
        h3_ids = [h3.geo_to_h3(lat, lon, resolution) for lat, lon in coords]

        # Assign H3 ID to each node in the graph
        for idx, h3_id in enumerate(h3_ids):
            g.vs[idx]['h3'] = h3_id

        # Step 2: Group original nodes by their H3 cell
        h3_data = {}
        for idx, h3_id in enumerate(h3_ids):
            if h3_id not in h3_data:
                h3_data[h3_id] = {
                    'nbAvis': [],
                    'typeR': []
                }
            h3_data[h3_id]['nbAvis'].append(nbAvis[idx])
            h3_data[h3_id]['typeR'].append(types[idx])

        # Step 3: Create a new H3-based graph
        unique_h3 = list(h3_data.keys())
        h3_to_index = {h3_id: idx for idx, h3_id in enumerate(unique_h3)}
        new_g = ig.Graph()
        new_g.add_vertices(len(unique_h3))
        new_g.vs['h3'] = unique_h3

        # Compute aggregated attributes for each H3 node
        lats = []
        lons = []
        mean_nbAvis = []
        mode_type = []
        group_size = []

        for h3_id in unique_h3:
            lat, lon = h3.h3_to_geo(h3_id)
            lats.append(lat)
            lons.append(lon)

            avis = h3_data[h3_id]['nbAvis']
            types = h3_data[h3_id]['typeR']

            mean_nbAvis.append(sum(avis) / len(avis))

            type_counter = Counter(types)
            most_common_type = type_counter.most_common(1)[0][0]
            mode_type.append(most_common_type)

            group_size.append(len(avis))

        # Set aggregated attributes on the new H3 graph
        new_g.vs['latitude'] = lats
        new_g.vs['longitude'] = lons
        new_g.vs['mode_nbAvis'] = mean_nbAvis
        new_g.vs['mode_typeR'] = mode_type
        new_g.vs['group_size'] = group_size

        # Step 4: Define edges between H3 nodes based on the original graph
        h3_edges = set()
        for e in g.es:
            h3_src = g.vs[e.source]['h3']
            h3_tgt = g.vs[e.target]['h3']
            if h3_src != h3_tgt:
                h3_edges.add(tuple(sorted((h3_src, h3_tgt))))

        edge_list = [(h3_to_index[src], h3_to_index[tgt]) for src, tgt in h3_edges]
        new_g.add_edges(edge_list)

        # Append the new graph to the list
        grouped_graphs.append(new_g)

    return grouped_graphs

# Parse gSpan output file into a list of igraph.Graph pattern objects.

def parse_gspan_patterns_h3(gspan_file_path):
    """
    Parse gSpan output file into a list of igraph.Graph pattern objects.
    """
    subgraphs = []
    current_graph = None
    vertex_labels = {}

    with open(gspan_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 't':
                # If a graph was being processed, add it to the list
                if current_graph is not None:
                    subgraphs.append(current_graph)

                # Start a new graph
                current_graph = ig.Graph()
                vertex_labels = {}

            elif parts[0] == 'v':
                # Add a new vertex
                vertex_id = int(parts[1])
                label = parts[2]
                current_graph.add_vertex(name=str(vertex_id), label=label)
                vertex_labels[vertex_id] = label

            elif parts[0] == 'e':
                # Add an edge between two vertices
                source = int(parts[1])
                target = int(parts[2])
                current_graph.add_edge(str(source), str(target))

    # Add the last graph if the file does not end with 't'
    if current_graph is not None:
        subgraphs.append(current_graph)

    return subgraphs


# Function for recovering and adding new metrics to the frequent vertices

def extract_vertices_patterns_ids_original_H3(output_filename_original_id, list_dataframes):

    with open(output_filename_original_id, 'r') as f:
        lines = f.readlines()

    L_patterns = []
    current_nodes = []

    for line in lines:
        line = line.strip()
        if line.startswith('t #'):
            #print(" ----- T " , line)
            #print(current_nodes)
            if current_nodes:
                L_patterns.append(processing_subgraph_ids(current_nodes, list_dataframes))
                current_nodes = []
        elif line.startswith('v'):
            #print(" ----- V ", line)
            parts = line.split()
            #print("parts :::: ", parts)
            if len(parts) >= 2: ### For getting the ID
            #if len(parts) >= 3: ### For getting the label
                #current_nodes.append(int(parts[1]))  # id original del nodo
                current_nodes.append(int(parts[2]))  # label original del nodo 
    if current_nodes:
        L_patterns.append(processing_subgraph_ids(current_nodes, list_dataframes))

    return L_patterns

# Function for building a dataframe with metrics and for different dates

def processing_subgraph_ids_h3(id_list, list_dataframes):
    rows = []
    ############print('ID_list -> ', id_list)
    for node_id in id_list:
        row = {'h3': node_id}
        for i, df in enumerate(list_dataframes):
            #print(" buscar el nodo ", node_id )
            #print(" en ", df.head() )
            df_match = df[df['h3'] == node_id]
            #print(" df match :::::: " , df_match)
            if not df_match.empty:
                #print (' ---> ', node_id)
                row[f'mode_nbAvis_{i}'] = df_match.iloc[0].get('mode_nbAvis', None)
                row[f'indegree_{i}'] = df_match.iloc[0].get('indegree', None)
                row[f'bcentrality_{i}'] = df_match.iloc[0].get('bcentrality', None)
                row[f'amenity_count_{i}'] = df_match.iloc[0].get('amenity_count', None)
                row[f'mca_huff_{i}'] = df_match.iloc[0].get('mca_huff', None)
            else:
                #row[f'name_{i}'] = "None"
                row[f'mode_nbAvis_{i}'] = "None"
                row[f'indegree_{i}'] = "None"
                row[f'bcentrality_{i}'] = "None"
                row[f'amenity_count_{i}'] = "None"
                row[f'mca_huff_{i}'] = "None"
        rows.append(row)

    #print(" procesar subgrafo ------ " , pd.DataFrame(rows).head())

    return pd.DataFrame(rows)

# Function for recovering original graph and buillding metrics from them and for each pattern

def extract_vertices_patterns_ids_original_H3(output_filename_original_id, list_dataframes):

    with open(output_filename_original_id, 'r') as f:
        lines = f.readlines()

    L_patterns = []
    current_nodes = []

    for line in lines:
        line = line.strip()
        if line.startswith('t #'):
            #print(" ----- T " , line)
            #print(current_nodes)
            if current_nodes:
                L_patterns.append(processing_subgraph_ids_h3(current_nodes, list_dataframes))
                current_nodes = []
        elif line.startswith('v'):
            #print(" ----- V ", line)
            parts = line.split()
            #print("parts :::: ", parts)
            if len(parts) >= 2: ### For getting the ID
            #if len(parts) >= 3: ### For getting the label
                #current_nodes.append(int(parts[1]))  # id original del nodo
                current_nodes.append(parts[2])  # label original del nodo 
    if current_nodes:
        L_patterns.append(processing_subgraph_ids_h3(current_nodes, list_dataframes))

    return L_patterns

# Function for ploring metrics for each H3

def plot_metrics_h3(df, metrics, output_folder="metrics_ot_h3"):
    """
    Plot metrics by pattern and by time (months).

    Args:
    df DataFrame: dataframe with metrics by nodes
    metrics list: a list of metrics should be ploted
    directory: the output directory storing all figures

    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for metric in metrics:
        columnas_mes = [f"{metric}_{i}" for i in range(len(metrics))]
        df_tmp = df[['h3'] + columnas_mes].set_index('h3')
        #names = df['name_0']

        df_tmp.fillna(0)
        #print(df_tmp)
        
        plt.figure(figsize=(10, 6))
        for place_id, fila in df_tmp.iterrows():
            plt.plot(range(1,6), fila.values, label=f"{place_id}")
        
        #plt.title(f"Evolución temporal de {metrica}")
        plt.xlabel("Months")
        plt.ylabel(metric)
        plt.xticks(range(1, 6), [f"Month {i}" for i in range(1, 6)])
        #plt.grid(True)
        plt.tight_layout()
        #plt.legend(names)
        plt.savefig(os.path.join(output_folder, f"{metric}_evolucion.png"))
        plt.close()


def count_freq_vertices_patt_h3(lst_nodes_patt, nodes_df):
    # Flatten the list of lists to count frequencies
    all_ids = [item for sublist in lst_nodes_patt for item in sublist]
    
    count = Counter(all_ids)
    
    # Convert the counter to a DataFrame
    df_frecuencia = pd.DataFrame(count.items(), columns=['h3', 'frequency'])
    
    # Join with the original dataframe to get the attributes
    df_result = pd.merge(df_frecuencia, nodes_df, on='h3', how='left')
    
    # Reordenar columnas si es necesario
    columnas_deseadas = ['h3', 'latitude', 'longitude', 'frequency']
    df_result = df_result[columnas_deseadas]
    
    return df_result

# Function for ploting a heatmap from pattern' H3 nodes frequency

def density_map_plotly_h3(df, center_lat=None, center_lon=None):

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
        hover_data={"frequency": True, "latitude": False, "longitude": False},
        height=600
    )

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

def count_amenities_by_h3(df):

    api = overpy.Overpass()

    lst_ammenities = [] 
    for index, row in df.iterrows():

        # Get the boundary of the H3 hexagon

        lat = row["latitude"]
        lon = row["longitude"]

        #print(lat, '----', lon)

        #h3_index = h3.h3_to_geo(row["h3"])
        #boundary = h3.cell_to_boundary(h3_index)
        boundary = h3.h3_to_geo_boundary(row["h3"])

        boundary_coords = [(lat, lng) for lat, lng in boundary]

        #polygon = Polygon(boundary_coords)

        overpass_coords = " ".join(f"{lat} {lon}" for lat, lon in boundary_coords)

        query = f"""
        [out:json];
        (
        node["hotel"](poly:"{overpass_coords}");
        );
        out center;
        """

        # Realiza la consulta a la API
        result = api.query(query)

        #print(result)

        count = len(result.nodes)

        lst_ammenities.append(count)

    # Map the counts back to the dataframe
    df['amenity_count'] = lst_ammenities

    return df

# Function for counting commodities in nodes of a iGraph

def count_amenities_by_h3_in_graph(graph):
    """
    For each node in the graph (with H3 index stored in 'label'), query the number of amenities 
    from Overpass API and add it as a new vertex attribute 'amenity_count'.

    Parameters:
    - graph (igraph.Graph): Graph with nodes having a 'label' attribute (H3 index).
    - tags_list (list of str): List of OSM tags to search (e.g., ["tourism", "hotel"]).
    - sleep_time (int): Seconds to wait between Overpass queries to avoid throttling.

    Returns:
    - igraph.Graph: The original graph with updated 'amenity_count' for each node.
    """
    api = overpy.Overpass()
    amenities_count = []

    for v in graph.vs:

        h3_index = v["h3"]
        api = overpy.Overpass()

        # Get the boundary of the H3 hexagon
        boundary = h3.h3_to_geo_boundary(h3_index)

        # Format the polygon for Overpass (in lat lon order)
        polygon_coords = " ".join([f"{lat} {lon}" for lat, lon in boundary])

        # Build the Overpass query
        query = f"""
        [out:json];
        (
            node["tourism"="hotel"](poly:"{polygon_coords}");
        );
        out body;
        """

        result = api.query(query)

        # Process the results
        count = len(result.nodes)

        amenities_count.append(count)

        #print(count)
    # Add the new 'amenity_count' attribute to the graph
    graph.vs["amenity_count"] = amenities_count

    return graph


def plot_h3_heatmap_folium(df, h3_column='h3', frequency_column='frequency', map_center=[50.643803,3.058299], zoom_start=12):
    # Initialize the Folium map
    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="cartodbpositron")

    # Normalize frequencies for coloring
    max_freq = df[frequency_column].max()

    for _, row in df.iterrows():
        h3_index = row[h3_column]
        frequency = row[frequency_column]

        # Get the boundary of the H3 cell (as list of [lat, lon])
        boundary = h3.h3_to_geo_boundary(h3_index)
        
        # Create a polygon for the H3 cell
        polygon = folium.vector_layers.Polygon(
            locations=boundary,
            color=None,
            fill=True,
            fill_color="red",
            fill_opacity=min(frequency / max_freq, 1.0)  # Normalize opacity
        )
        polygon.add_to(m)

    return m