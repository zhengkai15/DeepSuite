import torch 
from loguru import logger


def get_model(config):
    if config["model"]["name"] == "unet":
        fcst_steps = list(range(config["data"]["fcst_steps"]["beg"], config["data"]["fcst_steps"]["end"]+1, 1))
        best_model_path = config["exp"]["best_model_path"]
        in_varibales = 24
        in_times = len(fcst_steps)
        out_varibales = 1
        out_times = 72
        input_size = in_times * in_varibales
        output_size = out_times * out_varibales
        params = {'in_channels': input_size,
                'out_channels': output_size}
        
        # model = Model(params=params)

        # if best_model_path is not None:
        #     model.load_state_dict(torch.load(best_model_path))
            
    elif config["model"]["name"] == "temporal_model":
        # model = TemporalModel(ModelConfig(), configs=config)
        pass
    else:
        raise ValueError("Unsupported model name.")
    return None