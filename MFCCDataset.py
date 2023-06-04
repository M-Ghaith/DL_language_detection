import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T


# Data Transofrmation and Normalisation class
class MFCC(Dataset):
    def __init__(self, X, num_mfcc=13, n_fft=400, hop_length=160, sample_rate=8000):
        super(MFCC, self).__init__()
        self.X = X
        self.sample_rate = sample_rate

        self.transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=num_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'center': False,
            }
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Reshape data from (n,) to (1, n) and convert to tensor
        waveform = self.X[idx].clone().detach().reshape(1, -1)

        # Apply transform
        mfcc = self.transform(waveform)
        
        # Normalise MFCCs across the feature dimension
        mfcc = (mfcc - mfcc.mean(dim=1, keepdim=True)) / mfcc.std(dim=1, keepdim=True)
        
        return mfcc
