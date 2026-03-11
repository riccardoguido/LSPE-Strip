# LSPE-Strip

This repository collects material related to the data analysis activities carried out for the LSPE-Strip experiment.

The main directory contains:

- **`Thesis.pdf`**  
  The PDF version of the thesis.

- **`HDF5_files.xlsx`**  
  An Excel file containing the list of all HDF5 files with the acquisitions used in the data analysis, together with descriptions of the corresponding tests performed.

### `general/`

This folder contains general material about the LSPE-Strip instrument, including:

- **`Strip_cryostat.pdf`**  
  A document describing the components of the cryostat.

- **`Strip_FP.xlsx`**  
  An Excel file with tables describing the positions of the polarimeters and thermal sensors on the focal plane.

- **`Strip_polarimeters.xlsx`**  
  An Excel file reporting:
  - the correspondence between polarimeter names and their positions on the instrument tiles,
  - the default bias values (current and voltage) applied to each polarimeter.

### `codes/`

This folder contains the source code and notebooks developed for the data analysis:

- **`stripfunctions.py`**  
  Source file containing the functions developed for the data analysis.

- **`data_analysis.ipynb`**  
  A notebook providing a practical guide to the implemented functions and describing how they are used in the analysis work.

- **`spike_removal.ipynb`**  
  A notebook describing in detail the spike removal procedure applied to the data.


Additional material and documentation may be added to the repository in the future.
