from amrlib.models.parse_xfm.inference import Inference

model_dir = "Models/model_parse_xfm_bart_large"
base_model_path = "D:/Github/2025_Graph-based-Prototype/bart-large"

stog = Inference(model_dir, model_fn='pytorch_model.bin', tok_name=base_model_path)