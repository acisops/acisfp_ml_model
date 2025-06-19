from cheta import fetch_sci as fetch
from utils import split_data, FPTempDataProcessor, fields

msids = fetch.MSIDset(fields, "2020:001", "2025:108", stat="5min")

dp = FPTempDataProcessor()
times, data = dp.process_data(msids, fit=True)

ds = split_data(data, times)

dp.make_model()

dp.train_model(ds)
dp.save("dp.joblib")
