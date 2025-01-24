import pytorch_lightning as pl
from .dataset_constant import dataset_parameters
from .dataloader import load_data

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, train_loader, valid_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.test_mean = test_loader.dataset.mean
        self.test_std = test_loader.dataset.std
        self.data_name = test_loader.dataset.data_name

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader

    @property
    def num_train_samples(self):
        return len(self.train_loader.dataset)

    @property
    def num_val_samples(self):
        return len(self.valid_loader.dataset)

    @property
    def num_test_samples(self):
        return len(self.test_loader.dataset)



class WeatherLightningDataModule(object):
    def __init__(self, dataname, config):
        self.dataname = dataname
        self.config = config  # 修正：这里应该初始化self.config为传入的config，而不是None

    def get_data(self):
        train_loader, vali_loader, test_loader = self.get_dataset(self.dataname, self.config)
        return BaseDataModule(train_loader, vali_loader, test_loader)

    def get_dataset(self, dataname, config):
        config.update(dataset_parameters[dataname])
        return load_data(**config)

