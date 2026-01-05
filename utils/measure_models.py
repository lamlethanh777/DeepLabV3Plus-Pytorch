"""
Measure parameter count and FLOPs for different model variants.
"""
import torch
import network

def count_parameters(model):
    """Count trainable parameters in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def estimate_flops(model, input_size=(1, 3, 513, 513)):
    """
    Estimate FLOPs using a simple forward hook approach.
    For more accurate results, consider using thop or fvcore.
    """
    try:
        from thop import profile
        dummy_input = torch.randn(*input_size)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9  # Convert to GFLOPs
    except ImportError:
        # Fallback: rough estimation based on parameter count and typical compute patterns
        # This is a very rough approximation
        params = count_parameters(model)
        # Rough heuristic: ~2 FLOPs per parameter per pixel for CNNs
        h, w = input_size[2], input_size[3]
        return params * 2 * h * w / 1e3  # Very rough estimate

def main():
    # Model configurations to test
    models_config = [
        # (model_name, display_name, backbone)
        ("deeplabv3plus_mobilenet", "DeepLabV3+ (baseline)", "MobileNetV2"),
        ("deeplabv3plus_mobilenet_v3_large", "DeepLabV3+ (baseline)", "MobileNetV3-L"),
        ("deeplabv3plus_mobilenet_attention", "DeepLabV3+ + SA-ECA", "MobileNetV2"),
        ("deeplabv3plus_mobilenet_v3_large_attention", "DeepLabV3+ + SA-ECA", "MobileNetV3-L"),
        ("deeplabv3plus_mobilenet_epsa", "DeepLabV3+ + SP-EPSA", "MobileNetV2"),
    ]
    
    # mIoU results from experiments (VOC / Cityscapes)
    miou_results = {
        "deeplabv3plus_mobilenet": (66.65, 72.07),
        "deeplabv3plus_mobilenet_v3_large": (54.23, 63.74),
        "deeplabv3plus_mobilenet_attention": (67.96, 71.05),
        "deeplabv3plus_mobilenet_v3_large_attention": (54.17, 63.28),
        "deeplabv3plus_mobilenet_epsa": (65.45, 74.62),
    }
    
    print("=" * 100)
    print(f"{'Model':<35} {'Backbone':<15} {'Params (M)':<12} {'GFLOPs':<12} {'VOC mIoU':<12} {'City mIoU':<12}")
    print("=" * 100)
    
    results = []
    
    for model_name, display_name, backbone in models_config:
        try:
            # Create model
            model = network.modeling.__dict__[model_name](
                num_classes=21,  # VOC classes
                output_stride=16,
                pretrained_backbone=False  # Don't need pretrained for counting
            )
            model.eval()
            
            # Count parameters
            params = count_parameters(model)
            
            # Estimate FLOPs (using 513x513 for VOC, could also do 768x768 for Cityscapes)
            try:
                from thop import profile
                dummy_input = torch.randn(1, 3, 513, 513)
                flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
                gflops = flops / 1e9
            except ImportError:
                gflops = float('nan')
                print("Warning: thop not installed. Install with: pip install thop")
            
            voc_miou, city_miou = miou_results.get(model_name, (0, 0))
            
            print(f"{display_name:<35} {backbone:<15} {params:<12.2f} {gflops:<12.2f} {voc_miou:<12.2f} {city_miou:<12.2f}")
            
            results.append({
                'model': display_name,
                'backbone': backbone,
                'params': params,
                'gflops': gflops,
                'voc_miou': voc_miou,
                'city_miou': city_miou
            })
            
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
    
    print("=" * 100)
    
    # Generate LaTeX table
    print("\n\nLaTeX Table:\n")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Model complexity and performance comparison}")
    print(r"\label{tab:model_comparison}")
    print(r"\begin{tabular}{llcccc}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Backbone} & \textbf{Params (M)} & \textbf{GFLOPs} & \textbf{VOC mIoU} & \textbf{City mIoU} \\")
    print(r"\midrule")
    
    for r in results:
        print(f"{r['model']} & {r['backbone']} & {r['params']:.2f} & {r['gflops']:.2f} & {r['voc_miou']:.2f} & {r['city_miou']:.2f} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    main()
