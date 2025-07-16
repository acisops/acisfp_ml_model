from acisfp_ml_model import ACISFPMLModel


m = ACISFPMLModel.from_file("model_2020_2025.joblib")

m.make_web_page()
