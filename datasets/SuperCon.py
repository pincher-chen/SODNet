import numpy as np
import json
import os
import os.path as osp
import torch
import math

from io import StringIO
from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN,MinimumDistanceNN
from features.get_radius_graph_cutoff_knn import get_radius_graph_knn
from features.atom_feat import AtomCustomJSONInitializer
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm 


class SC(InMemoryDataset):

    def __init__(self, root, split,fold_data,fold_id,feature_type="crystalnet", fixed_size_split=True):
        assert feature_type in ["crystalnet"], "Please use valid features"
        assert split in ["train", "valid", "test"]
    
        self.split = split
        self.fold_data = fold_data
        self.fold_id = fold_id
        self.root = osp.abspath(root)
        self.feature_type = feature_type
        self.fixed_size_split = fixed_size_split

        super().__init__(self.root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def calc_stats(self, target):
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        y = y[:, target]
        mean = float(torch.mean(y))
        mad = float(torch.mean(torch.abs(y - mean))) #median absolute deviation
        return mean, mad

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def processed_file_names(self) -> str:
        return "_".join([self.split, self.feature_type]) +'_'+str(self.fold_id)+'.pt'

    def process(self):
        data_path ='./conf'
        ari = AtomCustomJSONInitializer(f'{data_path}/atom_embedding.json')
        r_cut = 8
        max_neighbors = 32
        data_list = []
        
        for i, cif_name in enumerate(tqdm(self.fold_data)):
            file_name = cif_name.split('/')[-1]
            all_data = open('datasets/SuperCon/df_all_data1202.csv','r')
            tc_data = all_data.readlines()
            
            for x in tc_data:
                cif_id = x.split(',')[0]
                tc = x.split(',')[1]
                if cif_id == file_name:
                    target = [math.log(float(tc))]
            y = torch.tensor(target)
            y = y.unsqueeze(0)
         
            crystal = Structure.from_file(cif_name)
            crystal = crystal.get_reduced_structure()
            crystal = crystal.get_primitive_structure()

            num_nodes = len(crystal.sites)

            atom_features = []
            occu_crystal = []
            for i in range(len(crystal.sites)):
                emb = 0
                total = 0
                for ele,occup in crystal[i].species.items():
                    num = ele.number
                    feature = np.vstack(ari.get_atom_features(num))
                    emb += feature*occup
                    total += occup
                atom_features.append(emb)
                occu_crystal.append(total)
               

            x = torch.tensor(atom_features).reshape((int(num_nodes), -1))
            edge_src, edge_dst, edge_vec, distances = get_radius_graph_knn(crystal,r_cut,max_neighbors)
            
            edge_occu = []
            
            for src,dst in zip(edge_src,edge_dst):
                 occu = occu_crystal[src]*occu_crystal[dst]
                 edge_occu.append(occu)
            
            distances = np.array(distances)
            name = file_name
             
            #build atom pairs within cutoff
            edge_num = len(edge_src)
            edge_num = torch.tensor(edge_num,dtype=torch.long)
            edge_src = torch.tensor(edge_src,dtype=torch.long)
            edge_dst = torch.tensor(edge_dst,dtype=torch.long)
            edge_occu = torch.tensor(edge_occu, dtype=torch.float)
            
            edge_vec = torch.tensor(edge_vec.astype(float), dtype=torch.float)
            edge_attr = torch.tensor(distances, dtype=torch.float)


            data = Data(x=x,edge_occu=edge_occu,edge_src=edge_src, edge_dst=edge_dst, 
                edge_attr=edge_attr,y=y, name=name, index=i,
                edge_vec = edge_vec, edge_num=edge_num)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])





