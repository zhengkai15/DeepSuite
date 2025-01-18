import torch 
from loguru import logger



def get_model(config):
    if config["model"]["name"] == "unet":
        from .models.unet import Model
        best_model_path = config["exp"]["best_model_path"]
        best_model_path = config["exp"]["best_model_path"]
        in_varibale_num = config["model"]["in_varibale_num"]
        in_times_num = config["model"]["in_times_num"]
        out_varibale_num = config["model"]["out_varibale_num"]
        out_time_num = config["model"]["out_time_num"]
        input_size = in_times_num * in_varibale_num
        output_size = out_time_num * out_varibale_num
        params = {'in_channels': input_size,
                'out_channels': output_size}
        
        model = Model(params=params)
        if best_model_path is not None:
            model.load_state_dict(torch.load(best_model_path))  
    elif config["model"]["name"] == "unet_yx_nowcasting":
        from .models.unet_yx_nowcasting import Model
        best_model_path = config["exp"]["best_model_path"]
        in_varibale_num = config["model"]["in_varibale_num"]
        in_times_num = config["model"]["in_times_num"]
        out_varibale_num = config["model"]["out_varibale_num"]
        out_time_num = config["model"]["out_time_num"]
        input_size = in_times_num * in_varibale_num
        output_size = out_time_num * out_varibale_num
        params = {'in_channels': input_size,
                'out_channels': output_size}
        model = Model(params=params)
        if best_model_path is not None:
            model.load_state_dict(torch.load(best_model_path))  
    elif config["model"]["name"] == "smaat_unet_yx_nowcasting":
        from .models.smaAt_unet import SmaAt_UNet as Model
        best_model_path = config["exp"]["best_model_path"]
        in_varibale_num = config["model"]["in_varibale_num"]
        in_times_num = config["model"]["in_times_num"]
        out_varibale_num = config["model"]["out_varibale_num"]
        out_time_num = config["model"]["out_time_num"]
        input_size = in_times_num * in_varibale_num
        output_size = out_time_num * out_varibale_num
        params = {'in_channels': input_size,
                'out_channels': output_size}
        model = Model(params=params)
        if best_model_path is not None:
            logger.info(f"best_model loaded:{best_model_path}")
            model.load_state_dict(torch.load(best_model_path))  
    elif config["model"]["name"] == "temporal_model":
        from .models.unet import Model
        # model = TemporalModel(ModelConfig(), configs=config)
        pass
    else:
        raise ValueError("Unsupported model name.")
    return model


