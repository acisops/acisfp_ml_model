from acisfp_ml_model import ACISFPMLModel

m = ACISFPMLModel()

msids = m.fetch_data("2020:001", "2025:108")
times, data = m.process_data(msids, fit=True)

ds = m.split_data(data, times)

m.make_model()

m.train_model(ds)

m.save("model_2020_2025.joblib")
