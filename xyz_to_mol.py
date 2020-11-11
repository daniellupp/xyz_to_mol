### This little script converts all the .xzy files (that do not already have a corresponding .mol file) to a .mol file
### It takes in coordinates, determines bonds and distances as well as the chirality of eligible atoms

import numpy as np
import pandas as pd
import re, os
from datetime import datetime

### The main class
class Molecule:
    def __init__(self, name, data, bond_length_dict):  ### Read the coordinates and put them in a suitable format
        self.name = name
        data = [n[:-1].split(" ") for n in data[2:]]
        atom_nrs = [n[0]+str(data.index(n)+1) for n in data]
        
        coordinates = [np.array(n[1:], dtype=float) for n in data]
        
        bond_length_df = pd.DataFrame(bond_length_dict).T
        bond_length_df.columns = ["min_length", "max_length", "bond_type"]
        bond_length_df["atoms"] = [n.split(" ")[0].split("-") for n in bond_length_df.index] 
        df = pd.DataFrame(coordinates, columns=["x","y","z"], index=atom_nrs, dtype=float)
        df_arrays = pd.DataFrame(dtype=float)
        df_arrays.loc[:, "arrays"] = coordinates
        df_arrays.index = atom_nrs

        Molecule.bond_dict = {}  ### Determine bond length, involved atoms and bond type and creates the corresponding Bond objects
        for atom1 in df.index:
            atom1_split = re.findall("([A-Za-z])([0-9]+)",atom1)[0]
            for atom2 in df.index:
                atom2_split = re.findall("([A-Za-z])([0-9]+)",atom2)[0]
                if int(atom1_split[1]) < int(atom2_split[1]):
                    distance = np.linalg.norm(df.loc[atom1,:]-df.loc[atom2,:])
                    for name in bond_length_df.index:
                        if sorted([atom1_split[0], atom2_split[0]]) == sorted(bond_length_df.at[name,"atoms"]):
                            if bond_length_df.at[name, "min_length"]<distance<bond_length_df.at[name, "max_length"]:
                                bond_name = atom1+"_"+atom2
                                Molecule.bond_dict[bond_name] = Bond(bond_name,atom1,atom2,distance,bond_length_df.at[name,"bond_type"])
        
        Molecule.atom_dict = {}  ### Determine atom type and chirality and creates the corresponding atom objects
        for atom in df.index:
            atom_split = re.findall("([A-Za-z]+)([0-9]+)", atom)
            atom_type, nr = atom_split[0][0], atom_split[0][1]
            Molecule.atom_dict[atom] = Atom(atom, atom_type, nr, Molecule.bond_dict, df.loc[atom,["x","y","z"]])  
        self.atom_names = [name for name in Molecule.atom_dict] 
        self.atoms = [Molecule.atom_dict.get(name) for name in self.atom_names] 
        for atom in self.atoms:
            if atom.atom_type == "C" and len(atom.bonds) == 4:
                prios = self.get_priorities(atom.atom_name, Molecule.atom_dict)
                if prios == ("no","t","chi","ral"):
                    continue
                else:
                    atom.stereogenic = self.determine_RS(prios, atom.atom_name, df_arrays)
    
    def bond_order_fix(self, frst_atom):  ### Determine the priorieties of the atoms and chains bonded to the tetrahedral carbon atom
        bond_order = ["single", "double", "triple"]
        prio_list = []
        for atm in frst_atom.bonded_atoms:
            bnd_nme = "_".join(sorted([frst_atom.atom_name,atm], key= lambda x: int(re.findall("([0-9]+)", x)[0])))
            multiplier = bond_order.index(Molecule.bond_dict[bnd_nme].bond_type)+1
            for n in range(multiplier):
                prio_list.append(atm)
        return(prio_list)
          
    def initial_priorities(self, atom):
        priorities = ["end","H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
        initial_prio = sorted([(priorities.index(re.findall("([A-Za-z]+)",a)[0])+1, a) for a in self.bond_order_fix(atom)], key = lambda x : x[0], reverse=True)
        return(initial_prio)
    
    def simple_priorities(self, atom_name_list):
        priorities = ["end","H", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"]
        prio = sorted([priorities.index(re.findall("([A-Za-z]+)",a)[0])+1 for a in atom_name_list], reverse=True)
        return(prio)
    
    def get_priorities(self, atom_name, atom_dict):
        
        atom = atom_dict.get(atom_name)      
        prio = self.initial_priorities(atom)
        prio_dict = {n[1]: n[0]for n in prio}
        step_dict = {}
        for m in [n[1] for n in prio if [n[0] for n in prio].count(n[0]) > 1]:
            step_dict[m] = [atom_name + "_" + m]
        while len(list(set([n[0] for n in prio]))) < 4:
            
            for n in step_dict:
                appendage=[]
                for l in step_dict[n]:
                   
                    curr_atom = l.rsplit("_",1)[-1]
                    if curr_atom == "end":
                        continue
                    nxt_atoms = [a[1] for a in self.initial_priorities(atom_dict.get(curr_atom)) if a[1] not in l.split("_")]
                    if nxt_atoms == []:
                        step_dict[n].append(l + "_end")
                        step_dict[n].remove(l)

                    else:
                        for nxt_atom in nxt_atoms:
                            appendage.append(l + "_" + nxt_atom)
                        step_dict[n].remove(l)
                step_dict[n].extend(appendage)        
            
            nxt_atoms_outer = []
            for n in step_dict:
                stems = list(set([cut.rsplit("_", 1)[0] for cut in step_dict[n]]))
                for stem in stems:
                    nxt_atoms_outer.append(((self.simple_priorities([path.rsplit("_",1)[1] for path in step_dict[n] if path.rsplit("_",1)[0] == stem]),n)))

            nxt_atoms_outer_prios_sorted = sorted(nxt_atoms_outer, key= lambda x: x[0], reverse=True)  
            if nxt_atoms_outer_prios_sorted[-1][0] < nxt_atoms_outer_prios_sorted[-2][0]:
                prio_dict[nxt_atoms_outer_prios_sorted[-1][1]] = float(prio_dict[nxt_atoms_outer_prios_sorted[-1][1]])-0.1
            if nxt_atoms_outer_prios_sorted[0][0] > nxt_atoms_outer_prios_sorted[1][0]:
                prio_dict[nxt_atoms_outer_prios_sorted[0][1]] = float(prio_dict[nxt_atoms_outer_prios_sorted[0][1]])+0.1
            if [n[0] for n in nxt_atoms_outer_prios_sorted].count([1]) >= 2:
                return(("no","t","chi","ral"))
                break 

            prio = sorted([(prio_dict[n], n) for n in prio_dict], key = lambda x: x[0], reverse=True)
        return(prio)

    def rot_matrix(self, angl, ax):  ### Determine R/S classification
        if ax == "x":
            return(np.array([[1,0,0],
                                [0, np.cos(angl),-np.sin(angl)],
                                [0,np.sin(angl),np.cos(angl)]]))
        if ax == "y":
            return(np.array([[np.cos(angl),0,np.sin(angl)],
                                [0, 1,0],
                                [-np.sin(angl),0,np.cos(angl)]]))
        if ax == "z":
            return(np.array([[np.cos(angl),-np.sin(angl),0],
                                [np.sin(angl),np.cos(angl),0],
                                [0,0,1]]))

    def determine_RS(self, prios, atom_name, df_arrays):
        ### move center atom to origin, rotate the 4th priority to the negative z axis(rot around x and y), 
        ### rotate 1st priority on the positive y axis, then going from the 1st to the 2nd in +x should be R, in negative should be S 
        df_arr_copy = df_arrays.copy()
        df_arr_copy["arrays"] = df_arr_copy.loc[:,"arrays"].apply(lambda x: x - df_arr_copy.at[atom_name, "arrays"])

        fourth_prio = df_arr_copy.at[prios[3][1],"arrays"]
        if fourth_prio[2] == float(0):
            angle = np.pi*0.5
        else:
            angle = np.arctan(fourth_prio[1]/fourth_prio[2])
            
        df_arr_copy["arrays"] = df_arr_copy.loc[:,"arrays"].apply(lambda x: np.matmul(self.rot_matrix(angle,"x"),x))

        fourth_prio = df_arr_copy.at[prios[3][1],"arrays"]
        if fourth_prio[2] == 0:
            angle = np.pi*0.5
        else:
            angle = -np.arctan(fourth_prio[0]/fourth_prio[2])
        if fourth_prio[2] < 0:
            angle = angle + np.pi
        df_arr_copy["arrays"] = df_arr_copy.loc[:,"arrays"].apply(lambda x: np.dot(self.rot_matrix(angle, "y"),x))

        fourth_prio = df_arr_copy.at[prios[3][1],"arrays"]
        if fourth_prio[2] > 0:
            angle = np.pi
        df_arr_copy["arrays"] = df_arr_copy.loc[:,"arrays"].apply(lambda x: np.dot(self.rot_matrix(angle, "y"),x))
        
        first_prio = df_arr_copy.at[prios[0][1],"arrays"]
        angle = -np.arctan(first_prio[1]/first_prio[0])
        df_arr_copy["arrays"] = df_arr_copy.loc[:,"arrays"].apply(lambda x: np.dot(self.rot_matrix(angle, "z"),x))
      
        first_prio = df_arr_copy.at[prios[0][1],"arrays"]
        if first_prio[0] < 0:
            angle = np.pi
            df_arr_copy["arrays"] = df_arr_copy.loc[:,"arrays"].apply(lambda x: np.dot(self.rot_matrix(angle, "z"),x))
        
        if df_arr_copy.at[prios[1][1],"arrays"][1] > 0:
            return("S")
        if df_arr_copy.at[prios[1][1],"arrays"][1] < 0:
            return("R")
    
    def zero_dep(self, e):
        if e<0:
            return("   ")
        else: 
            return("    ")
    
    def header_dep(self, nb):
        l = len(str(nb))
        if l<3:
            return((3-l)*" ")
        else: 
            return(" ")
        
    def output_mol(self):  ### Output the collected data in .mol specification
        name_line = f"{self.name}.xyz\n Dansxyztomol{datetime.now():%m%d%Y%H%M}3D"
        chirality = sum(set([1 if atom.stereogenic!=None else 0 for atom in self.atoms]))
        h1, h2 = len(self.atom_names), len(Molecule.bond_dict)
        header = f"\n{self.header_dep(h1)}{h1}{self.header_dep(h2)}{h2}  0  0  {chirality}  0  0  0  0  0999 V2000"
        zeros = "".join(["  0" for i in range(9)])
        coordinate_table = ["".join([f"{self.zero_dep(e)}{np.format_float_positional(e, precision=4, unique=False)}" 
                                         for e in atom.coordinates.tolist()])+" "+atom.atom_type for atom in self.atoms]
        chirality_12 = [0 if atom.stereogenic==None else 1 if atom.stereogenic=="R" else 2 for atom in self.atoms]
        zeros_table = [f"   0  0  {e}{zeros}" for e in chirality_12]
        atom_table = "\n".join([coordinate_table[i]+zeros_table[i] for i in range(len(zeros_table))])
        bonding_atoms = [f"  {re.findall('([0-9]+)',n.bonding_atom1)[0]} {re.findall('([0-9]+)',n.bonding_atom2)[0]}" 
                         for n in Molecule.bond_dict.values()]
        bonds = ["1" if n.bond_type=="single" else "2" if n.bond_type=="double" else "3" for n in Molecule.bond_dict.values()]
        zeros_2 = "".join([" 0" for i in range(4)])
        bond_table = "\n".join([f"{bonding_atoms[i]} {bonds[i]}{zeros_2}" for i in range(len(bonds))])
        output = f"{name_line}\n {header}\n{atom_table}\n{bond_table}\nM  END"
        return(output)
        
class Atom(Molecule):
    def __init__(self, atom_name, atom_type, nr, bond_dict, coordinates):
        self.atom_name = atom_name
        self.atom_type = atom_type
        self.nr = nr
        self.bonds = [bond_dict.get(bond) for bond in bond_dict if self.atom_name in bond.split("_")]
        self.coordinates = coordinates
        self.bonded_atoms = list(set([x for y in [bond.bond_name.split("_") for bond in self.bonds] for x in y]))
        self.bonded_atoms.remove(self.atom_name)
        self.stereogenic= None
        
class Bond:
    def __init__(self, bond_name,atom1, atom2, length, bond_type):
        self.bond_name = bond_name
        self.bonding_atom1 = atom1
        self.bonding_atom2 = atom2
        self.length = length
        self.bond_type = bond_type

bond_length_dict = {
    "H-H single": [0.64, 0.84, "single"], 
    "H-P single": [1.34, 1.54, "single"], 
    "H-S single": [1.24, 1.44, "single"], 
    "H-Si single": [1.38, 1.58, "single"], 
    "C-C single": [1.44, 1.64, "single"],
    "C-C double": [1.28, 1.34, "double"],
    "C-C triple": [1.1, 1.27, "triple"],
    "C-H single": [1.0, 1.2, "single"],
    "C-N single": [1.37, 1.57, "single"],
    "C-N double": [1.18, 1.36, "double"],
    "C-N triple": [1.06, 1.26, "triple"],
    "H-N single": [0.9, 1.1, "single"],
    "N-N single": [1.35, 1.55, "single"],
    "N-N double": [1.13, 1.34, "double"],
    "N-N triple": [1.0, 1.12, "triple"],
    "O-O single": [1.35, 1.55, "single"],
    "O-O double": [1.11, 1.34, "double"],
    "C-O single": [1.33, 1.53, "single"],
    "C-O double": [1.10, 1.30, "double"],
    "H-O single": [0.87, 1.17, "single"],
    "C-F single": [1.28, 1.48, "single"],
    "C-Cl single": [1.68, 1.88, "single"],
    "C-Br single": [1.84, 2.04, "single"],
    "C-I single": [2.04, 2.24, "single"],
    "N-O single": [1.26, 1.46, "single"],
    "N-O double": [1.1, 1.3, "double"],
    "S-S single": [1.97, 2.15, "single"],
    "S-S double": [1.8, 1.96, "double"],
    "S-O single": [1.54, 1.74, "single"],
    "S-O double": [1.39, 1.53, "double"],
    "S-N single": [1.67, 1.87, "single"],
    "S-N double": [1.44, 1.66, "double"],
    "S-C single": [1.68, 1.88, "single"],
    "S-C double": [1.5, 1.7, "double"],
    "C-P single": [1.74, 1.94, "single"]
    }


for x in [m.split(".")[0] for m in os.listdir('./') if not os.path.isfile(m.split(".")[0] + '.mol') if m.split(".")[-1]=="xyz"]:
    with open(f"{x}.xyz", "r") as file:
        data = file.readlines()            
    molcule = Molecule(x, data, bond_length_dict)
    with open(f"{x}.mol", "w") as file2:
        file2.write(molcule.output_mol())


























