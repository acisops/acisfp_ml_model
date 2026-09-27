from acisfp_ml_model import ACISFPMLModel


m = ACISFPMLModel.from_file("model_2020_2025.joblib")

outpath = "/proj/web-cxc/htdocs/acis/Thermal/acisfp_ml_model"

m.make_web_page(outpath=outpath)
