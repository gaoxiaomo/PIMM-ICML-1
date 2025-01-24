from weatherdata import WeatherLightningDataModule
from config import get_config
config = get_config()
data = WeatherLightningDataModule('weather_mv_4_4_s6_5_625',config).get_data()
x = data.train_dataloader()
print(data.train_dataloader())
print(data)