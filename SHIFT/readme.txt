SHIFT: Translating Human Mobility Forecasting
cfg/: config files
modules/: SHIFT: network model including two branches and the overall model; train.py: trainer file (directly used by cmd_train.py), used for inference with a trained model as well
data/: token: processed tokens using HuggingFace package; create_hybrid_dataset.py: data loader functions to load data for SHIFT
cmd_train.py: train model in cmd