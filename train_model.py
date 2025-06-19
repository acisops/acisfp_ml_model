from acisfp_model import ACISFPModel

dp = ACISFPModel()

msids = dp.fetch_data("2020:001", "2025:108")
times, data = dp.process_data(msids, fit=True)

ds = dp.split_data(data, times)

dp.make_model()

dp.train_model(ds)

dp.save("model_2020_2025.joblib")
