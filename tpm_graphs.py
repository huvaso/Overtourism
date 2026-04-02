#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Script for buiding graphs using iGraph and Networkit

# Basic Libraries
import numpy as np
import pandas as pd

# Libraries for buiding graphs (igraph is quickly than networkx)
import igraph as ig
from igraph import *

# Libraries for buiding graphs (networkit is quickly than igraph)
import networkit as nk

# Libraries for buiding graphs (networkx has many tools)
import networkx as nx

# Function for recovering CSV files
# input: paths to edges and vertices files
# output: two dataframes containing vertices and edges

def read_files(path_edges, path_vertices):
    #path_edges = "Locations/MEL_circulationGraph_0.csv"
    #path_vertices = "Locations/MEL_tripLocation.csv"
    vertices = pd.read_csv(path_vertices,sep="\t")
    edges = pd.read_csv(path_edges,sep=";") # change with "\t" for seasons graphs (instead ";")
    edges = edges.rename(columns={"gid_from":"source","gid_to":"target"})
    return vertices, edges

# Function for building a directed graph
# input: a dataframe of vertices and a dataframe of edges
# output: a directed graph

def building_graph_ig(vertices, edges):
    g = ig.Graph.DictList(
        vertices = vertices.to_dict('records'),
        edges = edges.to_dict('records'),
        directed = True,
        vertex_name_attr='id',
        #vertex_name_attr=vertices.index.values, 
        edge_foreign_keys=('source', 'target'));
    return g

# Function for getting a subgraph for the best values (percentage) of betweenness measure 
# input: a graph and a percentage
# output: a subgraph

def subgraph_btw(g, percentage):
    btwn = g.betweenness()
    ntile = np.percentile(btwn,percentage)
    g_small = g.vs.select([v for v, b in enumerate(btwn) if b >= ntile])
    gs = g.subgraph(g_small)
    return gs

# Function for deleting all isolated vertices and multiple arrays from and to the same vertices
# input: a graph
# output: the same graph without isolated vertices and simple edges
# Delete all self-loops edges like (0,0) and join all edges in one, e.g., (0,1) 18 is now (0,1) 1
# 18 means it exist 18 relations between vertices 0 and 1

def graph_simplify(g):
    g_sim = g.copy()
    return g_sim.simplify(loops=True, multiple=True)

# Function for extracting pairs of vertices participating in a edge
# input: the adjacency matrix of a graph
# output: a list of pairs (v1 v2)
# It is used for buiding a graph using the networkit lib

def extracting_pair_edges(adj_mat):
    pairs = []
    for i in range(len(adj_mat.nonzero()[0])):
        pairs.append((adj_mat.nonzero()[0][i],adj_mat.nonzero()[1][i]))
    return pairs

# Function for building a directed graph using networkit
# input: a graph in iGraph
# output: a directed graph in networkit

def building_graph_nk(g):
    adj_mat = g.get_adjacency_sparse()
    pairs = extracting_pair_edges(adj_mat)
    G = nk.Graph(len(g.vs), directed=True)
    for i in range(0,len(adj_mat.indices)):
        G.addEdge(pairs[i][0],pairs[i][1])
    return G

# Function for building a directed graph using networkx
# input: a graph in iGraph
# output: a directed graph in networkit

def building_graph_nx(g):
    adj_mat = np.array(g.get_adjacency().data)
    Gx = nx.from_numpy_matrix(np.matrix(adj_mat), create_using=nx.DiGraph())
    return Gx

# Function for comparing the vertices between the nodes in three graphs
# input: three graphs (gi, gx, gk)
# output: nothing

def compare_graphs_vertices(gi, gx, gk):
    nodes_gi = gi.vs.indices
    nodes_gx = list(gx.nodes)
    nodes_gk = list(gk.iterNodes())
    nodes_gi.sort() 
    nodes_gk.sort() 
    nodes_gx.sort()
    if nodes_gk == nodes_gi:
        print ("gk and gi are the same")
    if nodes_gx == nodes_gi:
        print ("gx and gi are the same")
    
# Function for comparing the adjacence marix of three graphs
# input: three graphs (gi, gx, gk)
# output: nothing

def compare_graphs_matrix_adj(gi, gx, gk):
    am_gk = nk.algebraic.adjacencyMatrix(gk).todense()
    am_gx = nx.adjacency_matrix(gx).todense()
    am_gi = np.matrix(gi.get_adjacency().data)
    if am_gk.all() == am_gi.all():
        print ("gk and gi are the same")
    if am_gx.all() == am_gi.all():
        print ("gx and gi are the same")

# Function for recovering CSV files for building SEASONS graph
# input: paths to edges and vertices files
# output: two dataframes containing vertices and edges

def read_files_season(path_edges, path_vertices):
    vertices = pd.read_csv(path_vertices,sep="\t")
    edges = pd.read_csv(path_edges,sep="\t") 
    edges = edges.rename(columns={"gid_from":"source","gid_to":"target"})
    return vertices, edges

# Function for building a directed graph for SEASONS
# input: a dataframe of vertices and a dataframe of edges
# output: a directed graph

def building_graph_ig_season(vertices, edges):
    g = ig.Graph.DictList(
        vertices = vertices.to_dict('records'),
        edges = edges.to_dict('records'),
        directed = True,
        vertex_name_attr='id',
        #vertex_name_attr=vertices.index.values, 
        edge_foreign_keys=('source', 'target'));
    return g