import torch
import torch.nn as nn
from torchvision.models import resnet50
from typing import List, Tuple, Dict
import pandas as pd
from collections import OrderedDict

def get_model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> pd.DataFrame:
    """
    Generate a detailed summary of a PyTorch model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
    
    Returns:
        DataFrame containing layer-wise model summary
    """
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__.__name__)
            module_idx = len(summary)
            
            # Get layer name
            m_key = f"{class_name}-{module_idx}"
            
            summary[m_key] = OrderedDict()
            summary[m_key]["Layer Type"] = class_name
            
            # Input shape
            if isinstance(input[0], (list, tuple)):
                summary[m_key]["Input Shape"] = str(list(input[0][0].size()))
            else:
                summary[m_key]["Input Shape"] = str(list(input[0].size()))
            
            # Output shape
            if isinstance(output, (list, tuple)):
                summary[m_key]["Output Shape"] = str(list(output[0].size()))
            else:
                summary[m_key]["Output Shape"] = str(list(output.size()))
            
            # Parameters
            params = 0
            trainable_params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                trainable_params += module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
                trainable_params += module.bias.requires_grad
            
            summary[m_key]["Params"] = params
            summary[m_key]["Trainable"] = trainable_params
    
        if not isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            hooks.append(module.register_forward_hook(hook))
    
    # Create model instance and move to evaluation mode
    model.eval()
    
    # Multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]
    
    # Batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size) for in_size in input_size]
    
    # Create properties
    summary = OrderedDict()
    hooks = []
    
    # Register hooks for all layers
    model.apply(register_hook)
    
    # Make a forward pass
    model(*x)
    
    # Remove these hooks
    for h in hooks:
        h.remove()
    
    # Create DataFrame
    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    
    # Calculate total parameters
    total_params = summary_df['Params'].sum()
    trainable_params = summary_df['Trainable'].sum()
    
    # Add total row
    summary_df.loc['Total'] = ['', '', '', total_params, trainable_params]
    
    return summary_df

def main():
    # Create ResNet50 model
    model = resnet50(pretrained=False)
    
    # Define input size (batch_size, channels, height, width)
    input_size = (3, 224, 224)
    
    # Generate and display summary
    summary_df = get_model_summary(model, input_size)
    
    # Print summary with formatting
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("\nResNet50 Model Summary:")
    print("=" * 80)
    print(summary_df)
    print("\nTotal Parameters:", summary_df.loc['Total', 'Params'])
    print("Trainable Parameters:", summary_df.loc['Total', 'Trainable'])

if __name__ == "__main__":
    main()