import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer
from torch import Tensor
from datasets import load_dataset
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader

def collate_fn(batch):
    x, y = zip(*batch)
    return torch.nn.utils.rnn.pad_sequence(x, batch_first=True), torch.cat(y)


class ImdbDataset(Dataset):
    
    def __init__(self, hf_dataset, tokenizer: PreTrainedTokenizer, max_len: int):
        super(ImdbDataset, self).__init__()
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index: int):
        return (
            Tensor(
                self.tokenizer.encode(
                    (self.data[index]["text"]).replace("<br/>", ""),
                    max_length=self.max_len,
                    truncation=True,
                )
            ).to(dtype=torch.long),
            Tensor([self.data[index]["label"]]).to(dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.data)
    
class ImdbDataModule(pl.LightningDataModule):
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int,
        num_workers: int,
        max_len: int,
    ):
        super(ImdbDataModule, self).__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len
        
    def prepare_data(self):
        pass
    
    def setup(self, stage: str) -> None:
        dataset = load_dataset("stanfordnlp/imdb", cache_dir="./data/")
        test = dataset["test"]
        test = test.train_test_split(test_size=0.5)
        dataset["test"] = test["train"]
        dataset["validation"] = test["test"]
        
        if stage == "fit" or stage is None:
            self.train_data = ImdbDataset(dataset["train"], self.tokenizer, self.max_len)
            self.val_data = ImdbDataset(dataset["validation"], self.tokenizer, self.max_len)
        else:
            self.test_data = ImdbDataset(dataset["test"], self.tokenizer, self.max_len)
    
    def test_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            shuffle=True
        )