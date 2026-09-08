# Training Guide

## Execution
```bash
python -m satellite_srm train --epochs 50
```

## Loss Function Formulation
The total optimization objective is:
$$L_{total} = \lambda_1 L_{L1} + \lambda_2 L_{perceptual} + \lambda_3 L_{spectral} + \lambda_4 L_{ndvi}$$

Where:
- $L_{L1}$: Masked L1 loss ignoring clouds/shadows
- $L_{perceptual}$: Deep VGG feature loss on RGB channels (B04, B03, B02)
- $L_{ndvi}$: Differentiable $|NDVI_{SR} - NDVI_{HR}|$ vegetation fidelity loss
- $L_{spectral}$: NIR/Red and Green/Red inter-band ratio preservation loss
