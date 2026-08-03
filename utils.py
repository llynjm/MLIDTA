import numpy as np
import torch
from math import sqrt
from scipy import stats
from torch_geometric import data as DATA
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.batch import Batch


class MoleculeFragData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "cluster_index":
            return int(value.max()) + 1
        if key == "frag_graph_edge_index":
            return int(self.cluster_index.max()) + 1
        if key == "mol_index":
            return 1
        return super().__inc__(key, value, *args, **kwargs)


class TestbedDataset(InMemoryDataset):
    def __init__(
        self,
        root="data",
        dataset="davis",
        xd=None,
        xt=None,
        smi=None,
        y=None,
        transform=None,
        pre_transform=None,
        smile_graph=None,
        target_key=None,
        target_graph=None,
    ):
        super().__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.process(xd, xt, smi, y, smile_graph, target_key, target_graph)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.dataset + ".pt"]

    def download(self):
        pass

    def process(self, xd, xt, smi, y, smile_graph, target_key, target_graph):
        assert len(xd) == len(xt) == len(y), "The three lists must be the same length."
        data_mol = []
        data_frags = []
        data_pro = []

        for i in range(len(xd)):
            smiles = xd[i]
            drug_smi_label = smi[i]
            target = xt[i]
            label = y[i]
            key = target_key[i]

            c_size, features, edge_index, fra_edge_index, cluster_index, frag_graph_edge_index = smile_graph[smiles]
            num_fragments = int(cluster_index.max()) + 1
            mol_index = torch.zeros((num_fragments,), dtype=torch.long)

            mol_data = DATA.Data(
                x=torch.tensor(features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                y=torch.tensor([label], dtype=torch.float),
            )
            mol_data.drug_smiles = torch.tensor([drug_smi_label], dtype=torch.long)
            mol_data.c_size = torch.tensor([c_size], dtype=torch.long)

            frag_data = MoleculeFragData(
                x=torch.tensor(features, dtype=torch.float),
                frags_edge_index=fra_edge_index,
                cluster_index=cluster_index,
                frag_graph_edge_index=frag_graph_edge_index,
                mol_index=mol_index,
            )

            p_size, p_features, p_edge_index = target_graph[key]
            pro_data = DATA.Data(
                x=torch.tensor(p_features, dtype=torch.float),
                edge_index=torch.tensor(p_edge_index, dtype=torch.long).t().contiguous(),
                y=torch.tensor([label], dtype=torch.float),
            )
            pro_data.target = torch.tensor([target], dtype=torch.long)
            pro_data.c_size = torch.tensor([p_size], dtype=torch.long)

            data_mol.append(mol_data)
            data_frags.append(frag_data)
            data_pro.append(pro_data)

        self.data_mol = data_mol
        self.data_frags = data_frags
        self.data_pro = data_pro
        print(f"Graph construction done: {len(self.data_mol)} samples.")

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_frags[idx], self.data_pro[idx]


def collate(data_list):
    mol_list = [data[0] for data in data_list]
    frags_list = [data[1] for data in data_list]
    pro_list = [data[2] for data in data_list]

    for i, frag_data in enumerate(frags_list):
        num_fragments = frag_data.cluster_index.max().item() + 1
        frag_data.mol_index = torch.full((num_fragments,), i, dtype=torch.long)

    return (
        Batch.from_data_list(mol_list),
        Batch.from_data_list(frags_list),
        Batch.from_data_list(pro_list),
    )


def rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))


def mse(y, f):
    return ((y - f) ** 2).mean(axis=0)


def pearson(y, f):
    return np.corrcoef(y, f)[0, 1]


def spearman(y, f):
    return stats.spearmanr(y, f)[0]


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    s = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z += 1
                u = f[i] - f[j]
                if u > 0:
                    s += 1
                elif u == 0:
                    s += 0.5
            j -= 1
        i -= 1
        j = i - 1
    return s / z


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    numerator = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean)) ** 2
    denominator = sum((y_obs - y_obs_mean) ** 2) * sum((y_pred - y_pred_mean) ** 2)
    return numerator / float(denominator)


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = np.mean(y_obs)
    numerator = sum((y_obs - (k * y_pred)) ** 2)
    denominator = sum((y_obs - y_obs_mean) ** 2)
    return 1 - (numerator / float(denominator))


def rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))
