import torch
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
from  rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm
import dgl
import pickle
import os
from collections import Counter


CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}


def label_smiles(line, smi_ch_ind, length):
    X = np.zeros(length,dtype=np.int64())
    for i, ch in enumerate(line[:length]):
        X[i] = smi_ch_ind[ch]
    return X


class CustomData(Data):
    '''
    Since we have converted the node graph to the line graph, we should specify the increase of the index as well.
    '''

    def __inc__(self, key, value, *args, **kwargs):
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "def __inc__(self, key, value, *args, **kwargs)"
        if key == 'line_graph_edge_index':
            return self.edge_index.size(1) if self.edge_index.nelement() != 0 else 0
        return super().__inc__(key, value, *args, **kwargs)
        # In case of "TypeError: __inc__() takes 3 positional arguments but 4 were given"
        # Replace with "return super().__inc__(self, key, value, args, kwargs)"
def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, atom_symbols, explicit_H=True, use_chirality=False):
    results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def edge_features(bond):
    bond_type = bond.GetBondType()
    return torch.tensor([
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]).long()

def generate_drug_data(mol_graph, atom_symbols, id):
    #get graph feature
    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)

    line_graph_edge_index = torch.LongTensor([])
    if edge_list.nelement() != 0:
        conn = (edge_list[:, 1].unsqueeze(1) == edge_list[:, 0].unsqueeze(0)) & (edge_list[:, 0].unsqueeze(1) != edge_list[:, 1].unsqueeze(0))
        line_graph_edge_index = conn.nonzero(as_tuple=False).T

    new_edge_index = edge_list.T
    #get Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol_graph, 2)
    fp = fp.ToBitString()
    fps = []
    for char in fp:
        fps.append(int(char))
    fps = np.array(fps)
    #get SMILES
    smile = Chem.MolToSmiles(mol_graph)
    smile_enc = label_smiles(smile, CHARISOSMISET, len(smile))
    if len(smile) >= 100:
        smile_enc = smile_enc[:100]
    else:
        smile_enc = np.pad(smile_enc, (0, (100 - len(smile_enc))), mode='constant')

    data = CustomData(x=features, edge_index=new_edge_index, line_graph_edge_index=line_graph_edge_index, edge_attr=edge_feats, fp=fps, id=id, smile_enc=smile_enc)

    return data

def generate_drug_data_dgl(mol_graph, atom_symbols):
    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), *edge_features(b)) for b in mol_graph.GetBonds()])
    edge_list, edge_feats = (edge_list[:, :2], edge_list[:, 2:].float()) if len(edge_list) else (torch.LongTensor([]), torch.FloatTensor([]))
    edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    edge_feats = torch.cat([edge_feats] * 2, dim=0) if len(edge_feats) else edge_feats

    features = [(atom.GetIdx(), atom_features(atom, atom_symbols)) for atom in mol_graph.GetAtoms()]
    features.sort()
    _, features = zip(*features)
    features = torch.stack(features)

    node_feature = features.long()
    edge_feature = edge_feats.long()

    g = dgl.DGLGraph()
    g.add_nodes(features.shape[0])
    g.ndata['feat'] = node_feature
    for src, dst in edge_list:
        g.add_edges(src.item(), dst.item())
    g.edata['feat'] = edge_feature
    data_dgl = g
    return data_dgl


def load_drug_mol_data(args):
    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    drug_id_mol_tup = []
    symbols = list()
    drug_smile_dict = {}
    smiles_rdkit_list = []

    for id1, id2, smiles1, smiles2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_s1],
                                                    data[args.c_s2], data[args.c_y]):
        drug_smile_dict[id1] = smiles1
        drug_smile_dict[id2] = smiles2

    for id, smiles in drug_smile_dict.items():
        #num = 0
        mol = Chem.MolFromSmiles(smiles.strip())

        if mol is not None:
            drug_id_mol_tup.append((id, mol))
            symbols.extend(atom.GetSymbol() for atom in mol.GetAtoms())

    for m in drug_id_mol_tup:
        smiles_rdkit_list.append(m[-1])#存的id
    symbols = list(set(symbols))
    drug_data = {id: generate_drug_data(mol, symbols, id) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_data')}
    save_data(drug_data, 'drug_data.pkl', args)
    drug_data_dgl = {id: generate_drug_data_dgl(mol, symbols) for id, mol in tqdm(drug_id_mol_tup, desc='Processing drugs_dgl')}
    save_data(drug_data_dgl, 'drug_data_dgl.pkl', args)
    return drug_data, drug_data_dgl


def generate_pair_triplets(args):
    pos_triples = []
    drug_ids = []

    with open(f'{args.dirname}/drug_data.pkl', 'rb') as f:
        drug_ids = list(pickle.load(f).keys())
        drug_ids = set(drug_ids)

    data = pd.read_csv(args.dataset_filename, delimiter=args.delimiter)
    for id1, id2, relation in zip(data[args.c_id1], data[args.c_id2], data[args.c_y]):
        relation -= 1
        pos_triples.append([id1, id2, str(relation)])  ## Turn to string for compatibility with id1 and id2 type

    if len(pos_triples) == 0:
        raise ValueError('Erroneous dataset! All tuples are invalid.')

    data_statistics = load_data_statistics(pos_triples)
    random_num_gen.shuffle(pos_triples)  # Shuffled in-place.

    old_old_triples = []
    new_old_triples = []
    new_new_triples = []
    old_drug_ids = []
    new_drug_ids = []
    secured_rels = []
    remaining_pos_trip = []

    for triple in pos_triples:
        if triple[2] in secured_rels:
            remaining_pos_trip.append(triple)
        else:
            secured_rels.append(triple[2])
            old_old_triples.append(triple)
            old_drug_ids.extend(triple[:2])

    num_new_drug = np.ceil(args.new_drug_ratio * len(drug_ids)).astype(int)
    total_num_drug = len(drug_ids)
    old_drug_ids = set(old_drug_ids)
    pos_triples = remaining_pos_trip
    remaining_drug_ids = list(drug_ids - old_drug_ids)

    random_num_gen.shuffle(remaining_drug_ids)

    new_drug_ids = set(remaining_drug_ids[:num_new_drug])
    old_drug_ids |= set(remaining_drug_ids[num_new_drug:])

    assert (new_drug_ids & old_drug_ids) == set()
    assert (new_drug_ids | old_drug_ids) == drug_ids

    for item in pos_triples:
        if (item[0] in new_drug_ids) and (item[1] in new_drug_ids):
            new_new_triples.append(item)
        elif (item[0] in old_drug_ids) and (item[1] in old_drug_ids):
            old_old_triples.append(item)
        else:
            new_old_triples.append(item)

    new_drug_ids = np.asarray(list(new_drug_ids))
    old_drug_ids = np.asarray(list(old_drug_ids))
    valid_drug_ids = (new_drug_ids, old_drug_ids)

    for pos_tups, desc in [
        (new_new_triples, 's2'),
        (old_old_triples, 'train'),
        (new_old_triples, 's1')
    ]:
        pos_tups = np.array(pos_tups)

        neg_samples = []
        for pos_item in tqdm(pos_tups, desc=f'Generating Negative sample for {desc}'):
            temp_neg = []
            h, t, r = pos_item[:3]

            if args.dataset == 'drugbank':
                neg_heads, neg_tails = _normal_batch(h, t, r, args.neg_ent, data_statistics, valid_drug_ids, desc)
                temp_neg = [str(neg_h) + '$h' for neg_h in neg_heads] + \
                           [str(neg_t) + '$t' for neg_t in neg_tails]
            else:
                raise NotImplementedError()
                existing_drug_ids = np.asarray(list(set(
                    np.concatenate(
                        [data_statistics["ALL_TRUE_T_WITH_HR"][(h, r)], data_statistics["ALL_TRUE_H_WITH_TR"][(h, r)]],
                        axis=0)
                )))
                temp_neg = _corrupt_ent(existing_drug_ids, args.neg_ent, valid_drug_ids)

            neg_samples.append('_'.join(map(str, temp_neg[:args.neg_ent])))

        df = pd.DataFrame({'Drug1_ID': pos_tups[:, 0],
                           'Drug2_ID': pos_tups[:, 1],
                           'Y': pos_tups[:, 2],
                           'Neg samples': neg_samples})
        filename = f'{args.dirname}/inductive/random_split/pair_pos_neg_triples-fold{args.seed}-{desc}.csv'
        df.to_csv(filename, index=False)
        print(f'\nData saved as {filename}!')
    save_data(data_statistics, 'data_statistics.pkl', args)


def load_data_statistics(all_tuples):
    '''
        This function is used to calculate the probability in order to generate a negative.
        You can skip it because it is unimportant.
        '''
    print('Loading data statistics ...')
    statistics = dict()
    statistics["ALL_TRUE_H_WITH_TR"] = defaultdict(list)
    statistics["ALL_TRUE_T_WITH_HR"] = defaultdict(list)
    statistics["FREQ_REL"] = defaultdict(int)
    statistics["ALL_H_WITH_R"] = defaultdict(dict)
    statistics["ALL_T_WITH_R"] = defaultdict(dict)
    statistics["ALL_TAIL_PER_HEAD"] = {}
    statistics["ALL_HEAD_PER_TAIL"] = {}

    for h, t, r in tqdm(all_tuples, desc='Getting data statistics'):
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)].append(h)
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)].append(t)
        statistics["FREQ_REL"][r] += 1.0
        statistics["ALL_H_WITH_R"][r][h] = 1
        statistics["ALL_T_WITH_R"][r][t] = 1

    for t, r in statistics["ALL_TRUE_H_WITH_TR"]:
        statistics["ALL_TRUE_H_WITH_TR"][(t, r)] = np.array(list(set(statistics["ALL_TRUE_H_WITH_TR"][(t, r)])))
    for h, r in statistics["ALL_TRUE_T_WITH_HR"]:
        statistics["ALL_TRUE_T_WITH_HR"][(h, r)] = np.array(list(set(statistics["ALL_TRUE_T_WITH_HR"][(h, r)])))

    for r in statistics["FREQ_REL"]:
        statistics["ALL_H_WITH_R"][r] = np.array(list(statistics["ALL_H_WITH_R"][r].keys()))
        statistics["ALL_T_WITH_R"][r] = np.array(list(statistics["ALL_T_WITH_R"][r].keys()))
        statistics["ALL_HEAD_PER_TAIL"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_T_WITH_R"][r])
        statistics["ALL_TAIL_PER_HEAD"][r] = statistics["FREQ_REL"][r] / len(statistics["ALL_H_WITH_R"][r])

    print('getting data statistics done!')

    return statistics


def _corrupt_ent(positive_existing_ents, max_num, drug_ids):
    corrupted_ents = []
    while len(corrupted_ents) < max_num:
        candidates = random_num_gen.choice(drug_ids, (max_num - len(corrupted_ents)) * 2, replace=False)
        invalid_drug_ids = np.concatenate([positive_existing_ents, corrupted_ents], axis=0)
        mask = np.isin(candidates, invalid_drug_ids, assume_unique=True, invert=True)
        corrupted_ents.extend(candidates[mask])

    corrupted_ents = np.array(corrupted_ents)[:max_num]
    return corrupted_ents


def _normal_batch(h, t, r, neg_size, data_statistics, drug_ids, desc):
    neg_size_h = 0
    neg_size_t = 0

    index = (0, 0) if desc == 's1' else (1, 1)
    if desc == 's2':
        index = (0, 1) if h in drug_ids[0] else (1, 0)

    prob = data_statistics["ALL_TAIL_PER_HEAD"][r] / (data_statistics["ALL_TAIL_PER_HEAD"][r] +
                                                      data_statistics["ALL_HEAD_PER_TAIL"][r])
    for i in range(neg_size):
        if random_num_gen.random() < prob:
            neg_size_h += 1
        else:
            neg_size_t += 1

    return (_corrupt_ent(data_statistics["ALL_TRUE_H_WITH_TR"][t, r], neg_size_h, drug_ids[index[0]]),
            _corrupt_ent(data_statistics["ALL_TRUE_T_WITH_HR"][h, r], neg_size_t, drug_ids[index[1]]))


def save_data(data, filename, args):
    dirname = f'{args.dirname}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    filename = dirname + '/' + filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f'\nData saved as {filename}!')


if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='drugbank' )
    parser.add_argument('-n', '--neg_ent', type=int, default=1, help='Number of negative samples')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the random number generator')
    parser.add_argument('-o', '--operation', type=str, default='all',
                        choices=['all', 'generate_triplets', 'drug_data', 'split'], help='Operation to perform')
    parser.add_argument('-n_d_r', '--new_drug_ratio', type=float, default=0.2)

    dataset_columns_map = {'drugbank': ('ID1', 'ID2', 'X1', 'X2', 'Y')}
    dataset_file_name_map = {
        'drugbank': ('data/drugbank.tab', '\t')
    }
    args = parser.parse_args()
    args.dataset = args.dataset.lower()

    args.c_id1, args.c_id2, args.c_s1, args.c_s2, args.c_y = dataset_columns_map[args.dataset]
    args.dataset_filename, args.delimiter = dataset_file_name_map[args.dataset]
    args.dirname = 'data'

    random_num_gen = np.random.RandomState(args.seed)
    # if args.operation in ('all', 'drug_data'):
    #     load_drug_mol_data(args)

    if args.operation in ('all', 'generate_triplets'):
        generate_pair_triplets(args)


