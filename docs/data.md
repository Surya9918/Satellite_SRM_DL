# Data Acquisition and Management

## Sentinel-2 Ingestion
Sentinel-2 L2A surface reflectance data is queried via the Copernicus Data Space Ecosystem (CDSE) OData catalog. Four bands are extracted:
- **B02** (Blue, 492.4 nm, 10m)
- **B03** (Green, 559.8 nm, 10m)
- **B04** (Red, 664.6 nm, 10m)
- **B08** (NIR, 832.8 nm, 10m)

## Reference Data (WorldStrat)
WorldStrat serves as an optical high-resolution reference proxy. Under CC-BY-4.0 terms, it provides paired observations for training and benchmarking.

## Geographic Region Splitting
To prevent spatial data leakage, adjacent patches from the same geographic tile are grouped into disjoint regional clusters:
- Train: 70%
- Validation: 15%
- Test: 15%
