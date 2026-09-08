# Full-Scene Tiled Inference

## Large-Scene Execution
```bash
python -m satellite_srm infer --input data/raw/sentinel2/scene.tif --output outputs/sr/SR_product.tif
```

## Overlap Blending
Scenes exceeding GPU memory are partitioned into overlapping tiles ($128 	imes 128$, overlap $16$ or $32$). Each tile is weighted using a 2D Hann window:
$$W(x, y) = \sin^2\left(rac{\pi x}{H}ight) \sin^2\left(rac{\pi y}{W}ight)$$
This eliminates edge seams and produces seamless mosaics.
