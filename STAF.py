import os
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.simplefilter("ignore")
from datetime import datetime
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import networkx as nx
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from node2vec import Node2Vec
import random
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import sys
from sklearn.model_selection import KFold

import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from transformers import EsmModel, EsmTokenizer
from transformers import EsmForMaskedLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
from sklearn.model_selection import train_test_split
import torch.optim as optim
from po2go.po2go import PO2GO  # 导入PO2GO模型
from po2go.model import load_model_checkpoint
from po2go.embedding_dataset import EmbeddingDataset
import pickle
from collections import OrderedDict

import scipy.spatial.distance as sd
from collections import defaultdict, deque
import math
from sklearn.decomposition import NMF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer

import subprocess
import tempfile
from pathlib import Path

from dgl.nn.pytorch import GATConv
import dgl
from DeepGO.models import DeepGOModel, DeepGOGATModel, MLPModel
from torch.cuda.amp import autocast

import gzip
import tempfile
from Bio.PDB import PDBParser
from Bio import SeqIO

import xgboost as xgb
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup
from scipy.sparse.csgraph import shortest_path
import torch.utils.checkpoint as checkpoint
from sklearn.ensemble import RandomForestClassifier

proteinlist = []
PPMatrix = np.empty_like([])
PGMatrix = np.empty_like([])
NList = []
GoList = []
PGONum = []
GPList = []
GOType = 'C'



def ordered_unique(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

def LoadData():
    global proteinlist
    TempList = []
    CAFAList = []
    CAFAFile = './data/cafa3_eukaryon.txt'
    PPIFile = './data/PPI_StringV12_min700.txt'
    print(CAFAFile, PPIFile)
    with open(CAFAFile, 'r') as file:
        for line in file:
            line = line.strip()
            beginstr, endstr, types = line.split('\t')
            if types != GOType:
                continue
            if beginstr not in TempList:
                TempList.append(beginstr)
    PList = []
    with open(PPIFile, 'r') as file:
        for line in file:
            line = line.strip()
            beginstr, endstr, scores, species = line.split('\t')
            if beginstr not in PList:
                PList.append(beginstr)
            if endstr not in PList:
                PList.append(endstr)
        list1 = ordered_unique(PList)
        list2 = ordered_unique(TempList)
        proteinlist = [item for item in list1 if item in list2]
        listlen = len(proteinlist)
        file.seek(0)
        global PPMatrix
        PPMatrix = np.zeros((listlen, listlen))
        global NList
        NList = [[0 for j in range(0)] for i in range(listlen)]

        for line in file:
            line = line.strip()
            beginstr, endstr, scores, species = line.split('\t')
            if (beginstr not in TempList) or (endstr not in TempList):
                continue
            Ipos = proteinlist.index(beginstr)
            JPos = proteinlist.index(endstr)
            PPMatrix[Ipos][JPos] = scores
            PPMatrix[JPos][Ipos] = scores
            NList[Ipos].append(JPos)
            NList[JPos].append(Ipos)

    with open(CAFAFile, 'r') as file:
        for line in file:
            line = line.strip()
            beginstr, endstr, types = line.split('\t')
            if types != GOType:
                continue
            if beginstr not in proteinlist:
                continue
            global GoList
            if endstr not in GoList:
                GoList.append(endstr)

    global PGMatrix
    PGMatrix = np.zeros((listlen, len(GoList)))
    global PGONum
    PGONum = [0 for i in range(listlen)]
    global GPList
    GPList = [[0 for j in range(0)] for i in range(len(GoList))]
    with open(CAFAFile, 'r') as file:
        for line in file:
            line = line.strip()
            beginstr, endstr, types = line.split('\t')
            try:
                Ipos = proteinlist.index(beginstr)
            except ValueError:
                Ipos = -1
            if types != GOType:
                continue
            if Ipos == -1:
                continue
            JPos = GoList.index(endstr)
            PGMatrix[Ipos][JPos] = 1
            PGONum[Ipos] = PGONum[Ipos] + 1
            GPList[JPos].append(Ipos)
def load_predicted_PDB(pdbfile):
    if pdbfile.endswith('.gz'):
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.pdb', delete=False) as tmp:
            with gzip.open(pdbfile, 'rb') as gz_file:
                tmp.write(gz_file.read())
            tmp_name = tmp.name
        parser = PDBParser()
        structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], tmp_name)
        records = SeqIO.parse(tmp_name, 'pdb-atom')
        os.unlink(tmp_name)
    else:
        parser = PDBParser()
        structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
        records = SeqIO.parse(pdbfile, 'pdb-atom')
    residues = [r for r in structure.get_residues()]
    seqs = [str(r.seq) for r in records]
    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one - two)
    return distances, seqs[0]


def get_valid_term_mask(PGMatrix, threshold=30):
    term_counts = np.sum(PGMatrix, axis=0)

    mask = term_counts >= threshold

    kept_num = np.sum(mask)
    total_num = len(mask)

    return mask
def GetESMSequenceFeature():
    # extract_ppi_protein_features("esm1b")
    ppi_data = np.load('./data/ppi_proteins_esm1b_features.npz')
    ppi_features = ppi_data['features']
    ppi_protein_ids = ppi_data['protein_ids']
    id_to_feature = {pid: feat for pid, feat in zip(ppi_protein_ids, ppi_features)}
    features = np.zeros((len(proteinlist), 1280))
    for i, pid in enumerate(proteinlist):
        if pid in id_to_feature:
            features[i] = id_to_feature[pid]
        else:
            print(f"warning: Protein: {pid} not found in Feature Library")
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    return {'features': normalized_features, 'protein_ids': np.array(proteinlist)}


def GetESM2_3B_Feature():
    # extract_ppi_protein_features("esm2_3b")
    ppi_data = np.load('./data/ppi_proteins_esm2_3b_features.npz')
    ppi_features = ppi_data['features']
    ppi_protein_ids = ppi_data['protein_ids']
    id_to_feature = {pid: feat for pid, feat in zip(ppi_protein_ids, ppi_features)}
    features = np.zeros((len(proteinlist), 2560))
    for i, pid in enumerate(proteinlist):
        if pid in id_to_feature:
            features[i] = id_to_feature[pid]
        else:
            print(f"warning: Protein: {pid} not found in Feature Library")
    return {'features': features, 'protein_ids': np.array(proteinlist)}


def read_fastasequence():
    sequences_dict = {}
    fasta_path = './data/uniprot_Sequence.fasta'
    with open(fasta_path, 'r') as f:
        current_id = None
        current_seq = []
        for line in f:
            if line.startswith('>'):
                if current_id is not None and current_id in proteinlist:
                    sequences_dict[current_id] = ''.join(current_seq)
                current_id = line[1:].strip().split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_id is not None and current_id in proteinlist:
            sequences_dict[current_id] = ''.join(current_seq)
    return sequences_dict





def main():
   
     LoadData()
     GetESMSequenceFeature()


if __name__ == "__main__":
    main()