import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Avalon import pyAvalonTools
from lstm_gnn.data import GetPubChemFPs
from torch_geometric.data import InMemoryDataset, DataLoader
import torch.nn.functional as F

from torch_geometric.nn import GATConv,GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import networkx as nx
from torch_geometric import data as DATA

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool

atts_out = []


class FP_LSTM(nn.Module):
    def __init__(self, args):
        super(FP_LSTM, self).__init__()
        self.fp_2_dim = args.fp_2_dim
        self.dropout_fpn = args.dropout
        self.cuda = args.cuda
        self.hidden_dim = args.hidden_size
        self.args = args

        self.fp_dim = 1489 + 512

        # self.embedding = nn.Linear(1,128)
        self.LSTM = nn.LSTM(1,256,5,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(512,1)
        self.fc1 = nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fpn)

    def forward(self, smile):
        fp_list = []
        for i, one in enumerate(smile):
            fp = []
            mol = Chem.MolFromSmiles(one)
            fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
            fp_phaErGfp = AllChem.GetErGFingerprint(
                mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
            fp_pubcfp = GetPubChemFPs(mol)
            fp_Avalon = pyAvalonTools.GetAvalonFP(mol)
            fp.extend(fp_maccs)
            fp.extend(fp_phaErGfp)
            fp.extend(fp_pubcfp)
            fp.extend(fp_Avalon)
            fp_list.append(fp)

        fp_list = torch.Tensor(fp_list)

        if self.cuda:
            fp_list = fp_list.cuda()
        fp_list = fp_list.unsqueeze(-1)
        # fpn_embedding = self.embedding(fp_list) # batch*len*1024
        fpn_lstm,(hn,cn) = self.LSTM(fp_list)
        fpn_list = self.fc(fpn_lstm)
        fp_list = fpn_list.squeeze()
        fpn_out = self.fc1(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        return fpn_out

class GNN_Dataset(InMemoryDataset):
    def __init__(self, smile):
        super(GNN_Dataset, self).__init__()
        data_list = []
        for item in smile:
            c_size, features, edge_index = self.smile_to_graph(item)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(
                                    edge_index).transpose(1, 0),
                                )
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)
        self.data, self.slices = self.collate(data_list)

    def atom_features(self, atom):
        return np.array(self.one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                        self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetHybridization(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self.one_of_k_encoding_unk(atom.GetChiralTag(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        [atom.GetIsAromatic()])

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception(
                "input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def smile_to_graph(self, smile):
        mol = Chem.MolFromSmiles(smile)

        c_size = mol.GetNumAtoms()

        features = []
        for atom in mol.GetAtoms():
            feature = self.atom_features(atom)
            features.append(feature / sum(feature))

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edges.append([bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])
        features = np.array(features)
        return c_size, features, edge_index

# GAT  model

class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=100, output_dim=300, dropout=0.2):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        return x


class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=100, output_dim=300, dropout=0.2):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd,
                            heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(dropout)

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=0.1, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        return x

class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=100, output_dim=300, dropout=0.2):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        return x

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=100, output_dim=300, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))

        return x


class GNN_Model(nn.Module):
    def __init__(self, dropout):
        super(GNN_Model, self).__init__()
        self.cuda = 'cuda:0'
        # self.model = GATNet(dropout=dropout)
        self.model = GAT_GCN(dropout=dropout)
        # self.model = GCNNet(dropout=dropout)
        # self.model = GINConvNet(dropout=dropout)

    def forward(self, smile):
        data = GNN_Dataset(smile)
        loader = DataLoader(data, batch_size=len(smile), shuffle=True)
        for batch_idx, data in enumerate(loader):
            data = data.to(self.cuda)
            output = self.model(data)
        return output


class LSTM_GNN_Model(nn.Module):
    def __init__(self, cuda, dropout_fpn):
        super(LSTM_GNN_Model, self).__init__()
        self.cuda = cuda
        self.dropout_fpn = dropout_fpn
        self.myenc = GNN_Model(dropout_fpn)
        self.sigmoid = nn.Sigmoid()

    def create_gat(self, args):
        # self.encoder3 = GAT(args)
        self.encoder3 = GNN_Model(args.dropout)

    def create_fpn(self, args):
        self.encoder2 = FP_LSTM(args)

    def create_scale(self, args):
        linear_dim = int(args.hidden_size)
        self.fc_gat = nn.Linear(linear_dim, linear_dim)
        self.fc_fpn = nn.Linear(linear_dim, linear_dim)
        self.act_func = nn.ReLU()

    def create_ffn(self, args):
        linear_dim = args.hidden_size
        self.ffn = nn.Sequential(
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=linear_dim*2,
                        out_features=linear_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(self.dropout_fpn),
            nn.Linear(in_features=linear_dim,
                        out_features=args.task_num, bias=True)
        )

    def forward(self, input):
        
        gat_out = self.encoder3(input)
        fpn_out = self.encoder2(input)
        gat_out = self.fc_gat(gat_out)
        gat_out = self.act_func(gat_out)

        fpn_out = self.fc_fpn(fpn_out)
        fpn_out = self.act_func(fpn_out)
        if len(fpn_out.shape)==1:
            fpn_out = fpn_out.unsqueeze(0)
        output = torch.cat([gat_out, fpn_out], axis=1)
        output = self.ffn(output)

        if not self.training:
            output = self.sigmoid(output)

        return output

def LSTM_GNN(args):
    model = LSTM_GNN_Model(args.cuda, args.dropout)
    model.create_gat(args)
    model.create_fpn(args)
    model.create_scale(args)
    model.create_ffn(args)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model
