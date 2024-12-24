import torch 
from loguru import logger



def get_model(config):
    if config["model"]["name"] == "unet":
        from .models.unet import Model
        best_model_path = config["exp"]["best_model_path"]
        in_varibale_num = 24
        in_times_num = 72
        out_varibale_num = 1
        out_time_num = 72
        input_size = in_times_num * in_varibale_num
        output_size = out_time_num * out_varibale_num
        params = {'in_channels': input_size,
                'out_channels': output_size}
        
        model = Model(params=params)
        if best_model_path is not None:
            model.load_state_dict(torch.load(best_model_path))      
    elif config["model"]["name"] == "temporal_model":
        from models.unet import Model
        # model = TemporalModel(ModelConfig(), configs=config)
        pass
    else:
        raise ValueError("Unsupported model name.")
    return model


