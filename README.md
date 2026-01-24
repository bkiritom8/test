# Cross-Platform PyTorch Training System

A production-ready PyTorch training framework designed to work seamlessly across **CUDA (NVIDIA GPUs)**, **MPS (Apple Silicon)**, and **CPU** backends with zero platform-specific code.

## Architecture Overview

This system consists of four main components:

### 1. **Device Management (`device_utils.py`)**
- Automatic device detection with priority: CUDA > MPS > CPU
- Cross-platform device operations (synchronization, memory management)
- Safe fallback mechanisms for unsupported operations
- Detailed device logging and memory statistics

### 2. **Neural Network Models (`model.py`)**
- `ConvolutionalClassifier`: Simple CNN for image classification
- `SimpleResNet`: Residual network with skip connections
- Both models are fully compatible with all backends
- Efficient architecture with minimal parameters

### 3. **Training Framework (`trainer.py`)**
- Universal training loop with automatic device placement
- Built-in validation and evaluation
- Checkpoint saving and loading
- Early stopping support
- Comprehensive metrics tracking
- Cross-platform memory management

### 4. **Main Application (`main.py`)**
- Command-line interface for easy configuration
- Synthetic data generation for demonstration
- Complete training pipeline with evaluation
- Configurable hyperparameters

## Features

- **Zero Platform-Specific Code**: Single codebase runs on all platforms
- **Automatic Device Detection**: Intelligent selection of best available accelerator
- **Robust Error Handling**: Graceful degradation and meaningful error messages
- **Memory Efficient**: Automatic cache clearing and memory tracking
- **Production Ready**: Logging, checkpointing, and monitoring built-in
- **Extensible**: Clean architecture for adding new models and features

## Installation

### Apple Silicon (MPS)

```bash
# Ensure you have Python 3.8+ installed
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with MPS support
pip install torch>=2.0.0

# Or install from requirements.txt
pip install -r requirements.txt
```

### Windows with NVIDIA GPU (CUDA)

```powershell
# Ensure you have Python 3.8+ installed
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Alternatively, visit https://pytorch.org to get the appropriate command
```

### CPU-Only (Any Platform)

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install PyTorch
pip install torch
```

## Usage

### Basic Training

```bash
# Auto-detect device and train for 10 epochs
python main.py

# Train for 20 epochs
python main.py --epochs 20

# Use ResNet model instead of CNN
python main.py --model resnet

# Force CPU mode (useful for testing)
python main.py --cpu
```

### Advanced Options

```bash
# Custom hyperparameters
python main.py \
    --epochs 30 \
    --batch-size 128 \
    --learning-rate 0.0001 \
    --model resnet

# Larger dataset
python main.py \
    --num-train 5000 \
    --num-val 1000 \
    --num-test 500

# Verbose logging
python main.py --verbose

# Custom checkpoint directory
python main.py --checkpoint-dir ./my_checkpoints
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cpu` | flag | False | Force CPU usage |
| `--epochs` | int | 10 | Number of training epochs |
| `--batch-size` | int | 64 | Batch size for training |
| `--learning-rate` | float | 0.001 | Learning rate |
| `--model` | str | cnn | Model architecture (cnn or resnet) |
| `--checkpoint-dir` | str | checkpoints | Checkpoint directory |
| `--verbose` | flag | False | Enable verbose logging |
| `--num-train` | int | 1000 | Number of training samples |
| `--num-val` | int | 200 | Number of validation samples |
| `--num-test` | int | 200 | Number of test samples |

## Platform-Specific Notes

### Apple Silicon (M1/M2/M3)

- MPS acceleration is automatically detected and enabled
- Requires PyTorch 2.0 or later for stable MPS support
- Some operations may fall back to CPU if not yet optimized for MPS
- Pin memory is disabled for MPS compatibility

### NVIDIA GPUs (CUDA)

- Automatic GPU selection if multiple GPUs available
- Memory statistics and monitoring enabled
- Cache clearing for optimal memory usage
- Full support for all PyTorch operations

### CPU Fallback

- Automatically used when no accelerator is available
- Warning logged to inform about potentially slower training
- Identical behavior to GPU modes (just slower)
- Useful for testing and development

## Code Structure

```
.
├── main.py              # Entry point and CLI
├── device_utils.py      # Device detection and management
├── model.py             # Neural network architectures
├── trainer.py           # Training and evaluation logic
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Extending the System

### Adding a New Model

```python
# In model.py
class MyCustomModel(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super(MyCustomModel, self).__init__()
        # Define your layers here

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define forward pass
        return x

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### Using Real Datasets

Replace the `create_data_loaders()` function in `main.py`:

```python
from torchvision import datasets, transforms

def create_data_loaders(batch_size: int = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    return train_loader, val_loader, test_loader
```

## Performance Tips

1. **Batch Size**: Increase batch size for better GPU utilization
2. **Data Loading**: Set `num_workers=0` for cross-platform compatibility
3. **Memory**: Monitor memory usage with `--verbose` flag
4. **Mixed Precision**: Can be added for CUDA GPUs (not supported on MPS yet)

## Troubleshooting

### MPS Issues

If MPS is detected but fails:
```bash
# Force CPU mode
python main.py --cpu
```

### CUDA Out of Memory

```bash
# Reduce batch size
python main.py --batch-size 32

# Reduce dataset size
python main.py --num-train 500
```

### Import Errors

```bash
# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"

# Reinstall if needed
pip install --upgrade torch
```

## Requirements

- Python 3.8 or later
- PyTorch 2.0 or later
- No platform-specific dependencies

## License

This is demonstration code for cross-platform PyTorch training.

## Contributing

When extending this code:
1. Test on multiple platforms (CUDA, MPS, CPU)
2. Avoid platform-specific imports or operations
3. Use device-agnostic PyTorch APIs
4. Add appropriate error handling and logging
