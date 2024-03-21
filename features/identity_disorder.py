import sys
sys.path.append(r"./")
import os
from pymatgen.core.structure import Structure

def identity_type(cif):
    crystal = Structure.from_file(cif)
    crystal = crystal.get_reduced_structure()
    crystal = crystal.get_primitive_structure()
    site_num = []
    all_occu = []
    struc_type = "order"
    if not crystal.is_ordered:
        for i in range(len(crystal.sites)):
            for el, occu in crystal[i].species.items():
                all_occu.append(occu)
            site_num.append(len(crystal[i].species))

        if max(site_num) > 1:
            struc_type = 'sd'
        else:
            struc_type = 'pd'
    return struc_type


def classify(dataset):  
    order_data,dis_data = [],[]
    for cif_file in dataset:
        stu_type=identity_type(cif_file)
        if stu_type == "order":
            order_data.append(cif_file)
        if stu_type == "sd" or stu_type == "pd":
            dis_data.append(cif_file)
    return order_data,dis_data      


