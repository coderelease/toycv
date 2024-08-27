import torch
from thop import profile


def get_model_statistics(model, inputs):
    """
    Calculate and print the FLOPs, MACs, and Params for a given model.

    Parameters:
    - model: The model to evaluate.
    - input_size: The size of the input tensor (batch_size, channels, height, width).

    Returns:
    - flops: Number of floating point operations.
    - macs: Number of multiply-accumulate operations.
    - params: Number of parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Calculate FLOPs and MACs
    # flops = FlopCountAnalysis(model, inputs).total()
    flops, params = profile(model, inputs=inputs)
    macs = flops / 2

    return params, flops, macs


def readable_number(number):
    """
    Convert a number to a human-readable string with appropriate units.

    Parameters:
    - number: The number to convert.

    Returns:
    - The human-readable string.
    """
    if number >= 1e12:
        return f"{number / 1e12:.3f} T"
    elif number >= 1e9:
        return f"{number / 1e9:.3f} G"
    elif number >= 1e6:
        return f"{number / 1e6:.3f} M"
    elif number >= 1e3:
        return f"{number / 1e3:.3f} K"
    else:
        return f"{number:.3f}"
