import torch 
from torch.utils.data import DataLoader
import os
import numpy as np 
# import pandas as pd
# from utils import MemmapGenome
# import pickle 
# import copy
# import pyBigWig
# import tabix 
# from selene_sdk.targets import Target


os.environ["MODEL_DIR"] = "./saved_models"  

" Toy Dataset Taken From Original Dirichelet Flow Matching for DNA Paper"
class ToyDataset(torch.utils.data.IterableDataset):
    def __init__(self, args, max_samples = 512000):
        super().__init__()
        self.num_cls = args.toy_num_cls
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim
        self.max_samples = max_samples  # Add termination condition


        self.probs = torch.softmax(torch.rand((self.num_cls, self.seq_len, self.alphabet_size)), dim=2)
        self.class_probs = torch.ones(self.num_cls)
        if self.num_cls > 1:
            self.class_probs = self.class_probs * 1 / 2 / (self.num_cls - 1)
            self.class_probs[0] = 1 / 2
        assert self.class_probs.sum() == 1

        distribution_dict = {'probs': self.probs, 'class_probs': self.class_probs}
        torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'toy_distribution_dict.pt' ))

    def __len__(self):
        #for some reason they have it ridiculously high (100, 000, 000), return to this later if there is an issue
        return self.max_samples  # Matches actual dataset size


    def __iter__(self):
        # original function just a while true loop with no stopping condition lol
        # this is a placeholder for now 
        count = 0
        while count < self.max_samples:  # Add termination condition
            cls = np.random.choice(a=self.num_cls, size=1, p=self.class_probs.numpy())
            seq = []
            for i in range(self.seq_len):
                seq.append(torch.multinomial(
                    self.probs[cls[0], i, :], 
                    num_samples=1, 
                    replacement=True
                ))
            yield torch.stack(seq).squeeze(-1), torch.tensor(cls)
            count += 1





# class EnhancerDataset(torch.utils.data.Dataset):
#     def __init__(self, args, split='train'):
#         all_data = pickle.load(open(f'data/the_code/General/data/Deep{"MEL2" if args.mel_enhancer else "FlyBrain"}_data.pkl', 'rb'))
#         self.seqs = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'{split}_data'])), dim=-1)
#         self.clss = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'y_{split}'])), dim=-1)
#         self.num_cls = all_data[f'y_{split}'].shape[-1]
#         self.alphabet_size = 4

#     def __len__(self):
#         return len(self.seqs)

#     def __getitem__(self, idx):
#         return self.seqs[idx], self.clss[idx]


class TwoClassOverfitDataset(torch.utils.data.IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim
        self.num_cls = 2

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'overfit_dataset.pt'))
            self.data_class1 = distribution_dict['data_class1']
            self.data_class2 = distribution_dict['data_class2']
        else:
            self.data_class1 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(args.toy_num_seq)])
            self.data_class2 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(args.toy_num_seq)])
            distribution_dict = {'data_class1': self.data_class1, 'data_class2': self.data_class2}
        torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'overfit_dataset.pt'))

    def __len__(self):
        return 10000000000

    def __iter__(self):
        while True:
            if np.random.rand() < 0.5:
                yield self.data_class1[np.random.choice(np.arange(len(self.data_class1)))], torch.tensor([0])
            else:
                yield self.data_class2[np.random.choice(np.arange(len(self.data_class2)))], torch.tensor([1])


# class GenomicSignalFeatures(Target):
#     """
#     #Accept a list of cooler files as input.
#     """

#     def __init__(self, input_paths, features, shape, blacklists=None, blacklists_indices=None,
#                  replacement_indices=None, replacement_scaling_factors=None):
#         """
#         Constructs a new `GenomicFeatures` object.
#         """
#         self.input_paths = input_paths
#         self.initialized = False
#         self.blacklists = blacklists
#         self.blacklists_indices = blacklists_indices
#         self.replacement_indices = replacement_indices
#         self.replacement_scaling_factors = replacement_scaling_factors

#         self.n_features = len(features)
#         self.feature_index_dict = dict(
#             [(feat, index) for index, feat in enumerate(features)])
#         self.shape = (len(input_paths), *shape)

#     def get_feature_data(self, chrom, start, end, nan_as_zero=True, feature_indices=None):
#         if not self.initialized:
#             self.data = [pyBigWig.open(path) for path in self.input_paths]
#             if self.blacklists is not None:
#                 self.blacklists = [tabix.open(blacklist) for blacklist in self.blacklists]
#             self.initialized = True

#         if feature_indices is None:
#             feature_indices = np.arange(len(self.data))

#         wigmat = np.zeros((len(feature_indices), end - start), dtype=np.float32)
#         for i in feature_indices:
#             try:
#                 wigmat[i, :] = self.data[i].values(chrom, start, end, numpy=True)
#             except:
#                 print(chrom, start, end, self.input_paths[i], flush=True)
#                 raise

#         if self.blacklists is not None:
#             if self.replacement_indices is None:
#                 if self.blacklists_indices is not None:
#                     for blacklist, blacklist_indices in zip(self.blacklists, self.blacklists_indices):
#                         for _, s, e in blacklist.query(chrom, start, end):
#                             wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = 0
#                 else:
#                     for blacklist in self.blacklists:
#                         for _, s, e in blacklist.query(chrom, start, end):
#                             wigmat[:, np.fmax(int(s) - start, 0): int(e) - start] = 0
#             else:
#                 for blacklist, blacklist_indices, replacement_indices, replacement_scaling_factor in zip(
#                         self.blacklists, self.blacklists_indices, self.replacement_indices,
#                         self.replacement_scaling_factors):
#                     for _, s, e in blacklist.query(chrom, start, end):
#                         wigmat[blacklist_indices, np.fmax(int(s) - start, 0): int(e) - start] = wigmat[
#                                                                                                 replacement_indices,
#                                                                                                 np.fmax(int(s) - start,
#                                                                                                         0): int(
#                                                                                                     e) - start] * replacement_scaling_factor

#         if nan_as_zero:
#             wigmat[np.isnan(wigmat)] = 0
#         return wigmat
    
    


# class PromoterDataset(torch.utils.data.Dataset):
#     def __init__(self, seqlength=1024, split="train", n_tsses=100000, rand_offset=0):
#         self.shuffle = False

#         class ModelParameters:
#             seifeatures_file = 'data/promoter_design/target.sei.names'
#             seimodel_file = 'data/promoter_design/best.sei.model.pth.tar'

#             ref_file = 'data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
#             ref_file_mmap = 'data/promoter_design/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap'
#             tsses_file = 'data/promoter_design/FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv'

#             fantom_files = [
#                 "data/promoter_design/agg.plus.bw.bedgraph.bw",
#                 "data/promoter_design/agg.minus.bw.bedgraph.bw"
#             ]
#             fantom_blacklist_files = [
#                 "data/promoter_design/fantom.blacklist8.plus.bed.gz",
#                 "data/promoter_design/fantom.blacklist8.minus.bed.gz"
#             ]

#             n_time_steps = 400

#             random_order = False
#             speed_balanced = True
#             ncat = 4
#             num_epochs = 200

#             lr = 5e-4

#         config = ModelParameters()

#         self.genome = MemmapGenome(
#             input_path=config.ref_file,
#             memmapfile=config.ref_file_mmap,
#             blacklist_regions='hg38'
#         )
#         self.tfeature = GenomicSignalFeatures(
#             config.fantom_files,
#             ['cage_plus', 'cage_minus'],
#             (2000,),
#             config.fantom_blacklist_files
#         )

#         self.tsses = pd.read_table(config.tsses_file, sep='\t')
#         self.tsses = self.tsses.iloc[:n_tsses, :]

#         self.chr_lens = self.genome.get_chr_lens()
#         self.split = split
#         if split == "train":
#             self.tsses = self.tsses.iloc[~np.isin(self.tsses['chr'].values, ['chr8', 'chr9', 'chr10'])]
#         elif split == "valid":
#             self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr10'])]
#         elif split == "test":
#             self.tsses = self.tsses.iloc[np.isin(self.tsses['chr'].values, ['chr8', 'chr9'])]
#         else:
#             raise ValueError
#         self.rand_offset = rand_offset
#         self.seqlength = seqlength

#     def __len__(self):
#         return self.tsses.shape[0]

#     def __getitem__(self, tssi):
#         chrm, pos, strand = self.tsses['chr'].values[tssi], self.tsses['TSS'].values[tssi], self.tsses['strand'].values[
#             tssi]
#         offset = 1 if strand == '-' else 0

#         offset = offset + np.random.randint(-self.rand_offset, self.rand_offset + 1)
#         seq = self.genome.get_encoding_from_coords(chrm, pos - int(self.seqlength / 2) + offset,
#                                                    pos + int(self.seqlength / 2) + offset, strand)

#         signal = self.tfeature.get_feature_data(chrm, pos - int(self.seqlength / 2) + offset,
#                                                 pos + int(self.seqlength / 2) + offset)
#         if strand == '-':
#             signal = signal[::-1, ::-1]
#         return np.concatenate([seq, signal.T], axis=-1).astype(np.float32)

#     def reset(self):
#         np.random.seed(0)

