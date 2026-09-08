# Scientific Limitations & Governance

1. **Reconstruction vs. Observation**: Sub-4m outputs are mathematically reconstructed representations synthesized by deep neural networks. They must NOT be claimed as physical, directly observed satellite imagery.
2. **WorldStrat Proxy**: WorldStrat is an open research reference proxy; atmospheric and seasonal variations mean it does not represent in-situ Indian ground truth.
3. **Uncertainty Quantification**: Monte Carlo variance highlights regions where model predictions exhibit epistemic instability (e.g. edge artifacts, unseen textures). It does not guarantee ground-truth physical error bounds.
4. **No-Hallucination Mandate**: The model is penalized against generating high-frequency textures that lack support in the low-resolution multispectral input.
