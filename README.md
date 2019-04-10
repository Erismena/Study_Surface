# 2d-fluids-analysis

## PIV
Raw PIV videos are read in as `.cine` files with [Python Image Sequence](https://github.com/openpiv) and are processed using [OpenPIV](https://github.com/openpiv).

Two main classes exist in the PIV processing/analysis pipeline. The first is `PIVDataProcessing`, which handles the creation of the numpy arrays of processed PIV data. The second is `PIVDataAnalysis`, which handles the analysis of the processed data and data extraction. An instance of each class is created for each PIV dataset worked with.

### Processing PIV data
To process PIV data, first create an instance of the `PIVDataProcessing` class with the `parent_folder` and `cine_name` specifiying the location of the raw data. Optional arguments specify attributes of the dataset:
- pre-processing measures `crop_lims` (`[top,bottom,left,right]`) with which to crop the image (BEFORE masking), a list of `maskers` with which to mask the frames AFTER cropping but BEFORE the PIV analysis
- PIV processing parameters `window_size`, `overlap`, `search_area_size`, `frame_diff` (how many frames between frame a and frame b)
- calibration parameters of the original `.cine` file `dx`, `dt`

To process the data, call the `.run_analysis` method. Optional arguments specify processing parameters:
- a list of frames (in the original `.cine`) to serve as the "a frames" `a_frames`
- a boolean `save`, with whether or not the results will be saved
- the signal-to-noise threshold `s2n_thresh` passed to `extended_search_area_piv`

New attributes are assigned to the data processing object as the analysis is run:
- `dt_ab` is the timestep between a and b frames
- `dt_frames` is the timestep between adjacent a frames, ASSUMING THIS IS A CONSTANT!

Assuming `save=True` was passed to `.run_analysis`, two files will be saved in `parent_directory` with varying filetype and name `cine_name`:
- the `PIVDataProcessing` object, pickled, with extension `.pkl`
- a 4-D numpy array with the analysis results, with velocities in pixels / frame, with extension `.npy`

To access the data, load the pickled `PIVDataProcessing` object and call the `.load_flowfield()` method. This will return an instance of the `PIVDataAnalysis` class, described below, in which velocity data is scaled given the `dx` and `dt` and PIV processing parameters. Alternatively, the `associate_flowfield()` method will store this instance in the `.data` field of the `PIVDataProcessing` class so there's just one object to pass around for each PIV dataset. The `.data` field is cleared from the `PIVDataProcessing` when the `.save()` method is called so as to not double-save the processed data (it is already saved as a `.npy` file!).

### Analyzing PIV data
The `PIVDataAnalysis` class stores the processed and scaled velocity data in the `.ff` (flowfield) attribute. Methods act as wrappers of the analysis functions defined in the `piv` module.
