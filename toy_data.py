import torch 
from torch.utils.data import DataLoader
import os
import numpy as np 

os.environ["MODEL_DIR"] = "./saved_models"  

" Toy Dataset Taken From Original Dirichelet Flow Matching for DNA Paper"
class ToyDataset(torch.utils.data.IterableDataset):
    def __init__(self, args, max_samples = 2000):
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