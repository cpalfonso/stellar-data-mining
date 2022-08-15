## STELLAR data mining scripts

Run `extract_subduction_data.py` with a configuration file (`subduction_config.yml` by default) to extract subduction kinematics, in addition to seafloor age, sediment thickness, and carbonate thickness, to a tabular format (default output filename is `subduction_data.csv`), ready for further exploration and processing.

Input data for the Clennett et al. (2020) plate model can be downloaded from the RDS using the `get_clennett_data.sh` shell script.
Different plate models or input datasets can be used by providing a different configuration file (using `subduction_config.yml` as a template).

### Requirements
- [gplately](https://github.com/GPlates/gplately)
- [pygplates](https://www.gplates.org/)
- scikit-image
- scikit-learn
- xarray
- joblib (for running in parallel)
