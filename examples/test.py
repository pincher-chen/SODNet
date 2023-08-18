import sys 
sys.path.append("..")
import numpy as np
import math
import torch
import torch.optim as optim
import warnings

from contextlib import suppress
from nets import model_entrypoint
from pymatgen.core import Structure
from features.get_radius_graph_cutoff_knn import get_radius_graph_knn
from features.atom_feat import AtomCustomJSONInitializer
from features.identity_disorder import identity_type
from inference import get_one_prediction
from engine import train_one_data

warnings.filterwarnings('ignore')
device = torch.device('cuda')

class Unittest():
    def __init__(self,cif):
        self.cif = cif
        self.test_process()
        self.test_model()
        self.test_predict() 
        self.test_identity_disorder()

    def test_process(self):
        ari = AtomCustomJSONInitializer('../conf/atom_embedding.json')
        crystal = Structure.from_file(self.cif)
        crystal = crystal.get_reduced_structure()
        crystal = crystal.get_primitive_structure()

        target=math.log(2.04)
        self.y = torch.tensor(target,device=device)
        self.y = self.y.unsqueeze(0)

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
        self.x = torch.tensor(atom_features,device=device).reshape((int(num_nodes), -1))
        r_cut = 5
        max_neighbors = 12

        edge_src, edge_dst, edge_vec, distances = get_radius_graph_knn(crystal,r_cut,max_neighbors)

        edge_occu = []
        for src,dst in zip(edge_src,edge_dst):
            occu = occu_crystal[src]*occu_crystal[dst]
            edge_occu.append(occu)

        distances = np.array(distances)
        edge_num = [len(edge_src)]
        self.edge_num = torch.tensor(edge_num,dtype=torch.long,device=device)
        self.edge_src = torch.tensor(edge_src,dtype=torch.long,device=device)
        self.edge_dst = torch.tensor(edge_dst,dtype=torch.long,device=device)
        self.edge_occu = torch.tensor(edge_occu, dtype=torch.float,device=device)
        self.edge_vec = torch.tensor(edge_vec.astype(float), dtype=torch.float,device=device)
        self.edge_attr = torch.tensor(distances, dtype=torch.float,device=device)
        self.batch = torch.zeros(self.x.shape[0],dtype=torch.long,device=device)    

    def test_model(self):
        create_model = model_entrypoint('graph_attention_transformer_nonlinear_l2_e3_noNorm')
        model = create_model(irreps_in='100x0e',
                             radius=5.0, num_basis=128,
                             out_channels=1,
                             task_mean=1,
                             task_std=0,
                             atomref=None,
                             drop_path=None)
        model = model.to(device)  
        parameters = model.parameters()
        opt_args = dict(lr=5e-5, weight_decay=0.01)
        optimizer = optim.AdamW(parameters, **opt_args)

        norm_factor = [1, 0]
        MAE = train_one_data(model=model, criterion=torch.nn.L1Loss(), norm_factor=norm_factor,
            x=self.x,y=self.y,batch=self.batch,edge_occu=self.edge_occu,edge_src=self.edge_src,
            edge_dst=self.edge_dst,edge_vec=self.edge_vec,edge_attr=self.edge_attr,edge_num=self.edge_num,
            device=device, epoch=1,optimizer=optimizer,amp_autocast = suppress)

    def test_predict(self):
        model = ('../best_models/1_save.pt')
        pred = get_one_prediction(model,self.x,self.batch,self.edge_occu,
                                  self.edge_src,self.edge_dst,self.edge_vec,self.edge_attr,self.edge_num)

    def test_identity_disorder(self):
        stu_type = identity_type(self.cif)

if __name__ == "__main__":
    Unittest('example_cif/1244.cif')
    print('Test Success')
