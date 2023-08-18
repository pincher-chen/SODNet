import numpy as np
import torch
from pymatgen.analysis.local_env import CrystalNN,MinimumDistanceNN

#defining periodic graph within cutoff.

def get_radius_graph_knn(structure, cutoff, max_neighbors):
    MNN = MinimumDistanceNN(cutoff=cutoff,get_all_sites=True)
    edge_src, edge_dest, edge_vec, distance = [], [], [], []
    
    for i in range(len(structure.sites)):
        start =i
        center_site = np.array(structure[i].coords)
        mdnn = MNN.get_nn_info(structure,i)
        atom_radius_i = []
        for elem_i, occu_i in structure[i].species.items():
             atom_radius_i.append(elem_i.atomic_radius)
        center_max_radius = max(atom_radius_i)
        
        for atom in mdnn:
            end = atom['site_index']
            end_coords = np.array(atom['site'].coords,dtype=object)
            atom_radius_j = []
            for elem_j, occu_j in atom['site'].species.items():
                atom_radius_j.append(elem_j.atomic_radius)
            neigh_max_radius = max(atom_radius_j)
            try:
                radius = center_max_radius + neigh_max_radius
            except:
                radius = 0
            if np.array(atom['site'],dtype=object)[1] < radius:
                continue
            edge_src += [start]
            edge_dest += [end]
            edge_vec_t = np.array(center_site) - np.array(end_coords)
            edge_vec.append(edge_vec_t)
            distance.append(np.array(atom['site'],dtype=object)[1])
    
    edge_src, edge_dest, edge_vec, edge_distances = np.array(edge_src), np.array(edge_dest), np.array(edge_vec), np.array(distance)   
    
    max_neigh_index = np.array([])
    ## KNN methods
    for i in range(len(structure.sites)):
        idx_i = (edge_src == i).nonzero()[0]
        distance_sorted = np.sort(edge_distances[idx_i])
        #To include self edge, not using max_neighbors -1 ;
        if len(distance_sorted) != 0:
            try:
                max_dist = distance_sorted[max_neighbors-1]
            except:
                max_dist = distance_sorted[-1]
            max_dist_index = np.where(edge_distances[idx_i] <= max_dist+0.001)
            max_dist_index = np.array(max_dist_index).flatten()

            max_neigh_index_t = [idx_i[i] for i in max_dist_index]
            max_neigh_index_t = np.array(max_neigh_index_t)
            max_neigh_index = np.append(max_neigh_index,max_neigh_index_t)

    max_neigh_index=max_neigh_index.flatten().astype(int)
    max_neigh_index=[max_neigh_index[i] for i in range(len(max_neigh_index))]

    edge_src, edge_dest, edge_vec, distances=edge_src[max_neigh_index], edge_dest[max_neigh_index], edge_vec[max_neigh_index], edge_distances[max_neigh_index]
    
    return edge_src, edge_dest, edge_vec, distances
