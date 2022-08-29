import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T
from project.datasets import LicensePlateDetectionDataset, LicensePlateRecognitionDataset


class BaseDataModule(pl.LightningDataModule):

    def __init__(self, root_dir, metadata_file) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.metadata_file = metadata_file 
        self.batch_size = 8
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=6)


    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=6)


    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=6)


class LicensePlateDetectionDataModule(BaseDataModule):


    def setup(self, stage=None) -> None:
        self.dataset = LicensePlateDetectionDataset(self.root_dir, self.metadata_file, self.transform)
        self.train_size = int(0.8 * len(self.dataset))
        self.val_size = int(0.15 * len(self.dataset))
        self.test_size = len(self.dataset) - self.train_size - self.val_size
        
        trainset, validset, testset = random_split(self.dataset, [self.train_size, self.val_size, self.test_size])
        self.trainset = trainset
        self.validset = validset
        self.testset = testset


class LicensePlateRecognitionDataModule(BaseDataModule):

    def __init__(self, root_dir, metadata_file) -> None:
        super().__init__(root_dir, metadata_file)

        self.batch_size = 1

    def setup(self, stage=None) -> None:
        self.dataset = LicensePlateRecognitionDataset(self.root_dir, self.metadata_file, self.transform)
        self.train_size = int(0.8 * len(self.dataset))
        self.val_size = int(0.15 * len(self.dataset))
        self.test_size = len(self.dataset) - self.train_size - self.val_size
        
        trainset, validset, testset = random_split(self.dataset, [self.train_size, self.val_size, self.test_size])
        self.trainset = trainset
        self.validset = validset
        self.testset = testset