import numpy as np

def make_vgg_config(variant: str) -> list:
    """
    Return the layer configuration for a VGG variant.
    """
    # Your implementation here
    name = variant.lower()

    cfgs = {
        "vgg11": [1, 1, 2, 2, 2],
        "vgg13": [2, 2, 2, 2, 2],
        "vgg16": [2, 2, 3, 3, 3],
        "vgg19": [2, 2, 4, 4, 4],
    }

    channel_counts = [64, 128, 256, 512, 512]

    config = []
    block_architecture = cfgs[name]

    for i in range(len(block_architecture)):
        num_convs = block_architecture[i]
        num_channel = channel_counts[i] 

        for _ in range(num_convs):
            config.append(num_channel)

        config.append('M')
        
    return config 
        
        