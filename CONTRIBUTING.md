# Contributing to Satellite-SRM

Thank you for your interest in contributing to the Deep Learning Based Super Resolution Mapping (SRM) project for Sentinel-2 satellite imageries.

## Code Standards
- Adhere to PEP 8 guidelines.
- Ensure all functions include complete docstrings with type annotations.
- Every commit must pass all unit and integration tests.
- Maintain geospatial coordinate system preservation and multi-band (B02, B03, B04, B08) spectral consistency.

## Pull Request Process
1. Create a feature branch from `main`.
2. Ensure `python -m unittest discover tests` passes with zero regressions.
3. Submit a Pull Request documenting changes, verification results, and any new configuration parameters.
