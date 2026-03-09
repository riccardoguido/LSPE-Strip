# Libraries

import striptease
from striptease import DataStorage, DataFile
from striptease import STRIP_BOARD_NAMES
from striptease import BOARD_TO_W_BAND_POL
from striptease import normalize_polarimeter_name
from striptease import get_polarimeter_index
from striptease import polarimeter_iterator
from striptease.biases import InstrumentBiases
from striptease import spectrum

from importlib import reload
reload(spectrum)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import scipy.stats
import astropy
from astropy.io import ascii
from astropy.time import Time
from tqdm import tqdm
from scipy.optimize import curve_fit
import json

import pdb
import pickle
import os
import time
import re
import datetime
import math

import h5py
import sqlite3

#--------------------------------------------------------------------------------------------------------------------------------

def database(ds_path):
    
    """
    Initialize a DataStorage object from the specified path.

    Args:
        ds_path: Path to the directory containing the HDF5 data files.

    Returns:
        ds: Database containing the data files.
    """
    
    ds = DataStorage(ds_path)
    
    return ds


def list_of_files(ds):
    
    """
    Prints the list of data files available in the database with their index and time range.

    Args:
        ds: Database containing the data files.
    """
    
    # Load all data files in the time range
    files = ds.get_list_of_files()
    
    for i, file in enumerate(files, start=1):
        print(f"file {i}: {file.path}")
        print(f"mjd_range: {file.mjd_range}")
        print()

        
def list_of_tags(ds, mjd_range):
    
    """
    Extract and print all tags available in the database within a given time range.

    Args:
        ds: Database containing the data files.
        mjd_range: Time range in Modified Julian Date (MJD) format.

    Returns:
        tags: List of tags, each with start time, end time, and name.
    """
    
    # Load all tags in the time range
    tags = ds.get_tags(mjd_range) 
    
    for tag in tags:
        print(f"(mjd_start = {tag.mjd_start}, mjd_end = {tag.mjd_end}, name = {tag.name})")
    
    return tags 


# Define board names to list of polarimeters
board_name = {
    "R": ["R0", "R1", "R2", "R3", "R4", "R5", "R6", "W3"],
    "V": ["V0", "V1", "V2", "V3", "V4", "V5", "V6", "W4"],
    "G": ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "W6"],
    "B": ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "W5"],
    "Y": ["Y0", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "W1"],
    "O": ["O0", "O1", "O2", "O3", "O4", "O5", "O6", "W2"],
    "I": ["I0", "I1", "I2", "I3", "I4", "I5", "I6"],
}

def list_of_polarimeters(polarimeters, Q_band=True, W_band=True):
    
    if isinstance(polarimeters, str):
        polarimeters = [polarimeters]

    pol_names = []

    for p in polarimeters:
        if p in board_name:
            pol_list = board_name[p]
            if not Q_band:
                pol_list = [x for x in pol_list if not x.startswith(p)]
            if not W_band:
                pol_list = [x for x in pol_list if not x.startswith("W")]
            pol_names.extend(pol_list)
        else:
            pol_names.append(p)

    return list(dict.fromkeys(pol_names))

#--------------------------------------------------------------------------------------------------------------------------------
# Reading data saved in HDF5 files on Database

def scientific_output(ds, mjd_range, data_type, polarimeters, detectors, Q_band=True, W_band=True):
    
    """
    Extract scientific output data for a list of polarimeters and detectors, within a given time range.
    
    Args:
        ds: Database containing the data files.
        mjd_range: Time range in Modified Julian Date (MJD) format.
        data_type: Type of data to load ('DEM', 'PWR').
        polarimeters: Polarimeters to consider (can be both a list of polarimeters e.g. ['R0', ...] and boards e.g. ['R', ...]).
        detectors: Detectors to consider (e.g. ['Q1', 'Q2', 'U1', 'U2'] or a subset).
        Q_band: If True consider Q-band polarimeters (default value: True).
        W_band: If True consider W-band polarimeters (default value: True).
        
    Returns:
        output_diz: Dictionary structured as {'polarimeter': {'detector': {'time', 'data'}}}.
    """
    
    # Identify all the polarimeters in input
    pol_names = list_of_polarimeters(polarimeters, Q_band, W_band)
    
    if isinstance(detectors, str):
        detectors = [detectors]
            
    # Final dictionary containing scientific output
    output_diz = {}  
    
    # Iterate over each polarimeter
    for pol in tqdm(pol_names, desc="Loading data"):
        output_diz[pol] = {}

        # Load data in the time range
        time, data = ds.load_sci(mjd_range, pol, data_type)
        time = Time(time, format="mjd").unix # Convert time to seconds
        
        # Extract data from each detector
        for dets in detectors:
            det = f"{data_type}{dets}"
            output = data[det]
            output = output.astype(float) # Convert data to float type

            # Skip if data is empty or contains only invalid values
            if output.size == 0 or np.all(np.isnan(output)): 
                continue

            # Save the results in the dictionary
            output_diz[pol][det] = {
                "time": time, # Timestamp in sec
                "data": output, # Scientific output
            }

    return output_diz


def housekeeping_data(ds, mjd_range, parameters, polarimeters, Q_band=True, W_band=True):
    
    """
    Extract housekeeping parameters data for a list of polarimeters, within a given time range.
    
    Args:
        ds: Database containing the data files.
        mjd_range: Time range in Modified Julian Date (MJD) format.
        parameters: Parameters to consider (e.g. ['VD0_HK', 'ID0_HK', 'VPIN0_HK', ...]).
        polarimeters: Polarimeters to consider (can be both a list of polarimeters e.g. ['R0', ...] and boards e.g. ['R', ...]).
        Q_band: If True consider Q-band polarimeters (default value: True).
        W_band: If True consider W-band polarimeters (default value: True).
        
    Returns:
        data_diz: Dictionary structured as {'polarimeter': {'parameter': 'time', 'data'}}}.
    """
    
    # Identify all the polarimeters in input
    pol_names = list_of_polarimeters(polarimeters, Q_band, W_band)
    
    if isinstance(parameters, str):
        parameters = [parameters]
    
    # Final dictionary containing scientific output
    data_diz = {}
        
    # Iterate over each polarimeter
    for pol in tqdm(pol_names, desc="Loading data"):
        data_diz[pol] = {}
        
        subgroup = f"POL_{pol}"
        
        # Iterate over each parameter
        for par in parameters:
            # Try to load the parameter first from the 'BIAS' group, if fail try from the 'DAQ' group
            for group in ["BIAS", "DAQ"]:
                try:
                    # Load data in the time range
                    time, data = ds.load_hk(mjd_range, group, subgroup, par)
                    data = data.astype(float) # Convert data to float type
     
                    # Skip if data is empty or contains only invalid values
                    if data.size == 0 or np.all(np.isnan(data)):
                        continue
                        
                    time = Time(time, format="mjd").unix  # Convert time to seconds
                    
                    # Save the results in the dictionary
                    data_diz[pol][par] = {
                        "time": time, # Timestamp in sec
                        "data": data, # Housekeeping data
                    }
                    break
                    
                except Exception:
                    continue

    return data_diz


def thermal_data(ds, mjd_range, sensor_name, get_raw=False):
    
    """
    Extract temperature values for a list of sensors, within a given time range.
    
    Args:
        ds: Database containing the data files.
        mjd_range: Time range in Modified Julian Date (MJD) format.
        sensor_name: Thermal sensors to consider (e.g. ['TS-CX1-Module-R', 'EX-CX18-SpareCx', ...]).
        get_raw: If True get raw values, if False get calibrated values [K] (default value: False).
        
    Returns:
        data_diz: Dictionary structured as {'sensor': {'time', 'data'}}}.
    """
    
    # Final dictionary containing temperature values
    data_diz = {}
        
    # Iterate over each thermal sensor
    if isinstance(sensor_name, str):
        sensor_name = [sensor_name]

    for sensor in sensor_name:
        # Load data in the time range
        time, data = ds.load_cryo(mjd_range, sensor, get_raw)

        # Skip if data is empty or contains only invalid values
        if data.size == 0 or np.all(np.isnan(data)):
            continue
        
        time = Time(time, format="mjd").unix  # Convert time to seconds

        # Save the results in the dictionary
        data_diz[sensor] = {
            "time": time, # Timestamp in sec
            "data": data, # Temperature values
        }

    return data_diz

#--------------------------------------------------------------------------------------------------------------------------------
# Computing scientific data

def scientific_data(ds, mjd_range, data_type, polarimeters, detectors, Q_band=True, W_band=True):
    
    """
    Process scientific output data: for DEM data computes double demodulated data; for PWR data computes total power data.

    Args:
        ds: Database containing the data files.
        mjd_range: Time range in Modified Julian Date (MJD) format.
        data_type: Type of data to load ('DEM', 'PWR').
        polarimeters: Polarimeters to consider (can be both a list of polarimeters e.g. ['R0', ...] and boards e.g. ['R', ...]).
        detectors: Detectors to consider (e.g. ['Q1', 'Q2', 'U1', 'U2'] or a subset).
        Q_band: If True consider Q-band polarimeters (default value: True).
        W_band: If True consider W-band polarimeters (default value: True).

    Returns:
        data_diz: Dictionary structured as {'polarimeter': {'detector': {'time', 'data'}}}.
    """
    
    # Load data for all polarimeters in the time range using scientific_output function
    output_diz = scientific_output(ds, mjd_range, data_type, polarimeters, detectors, Q_band, W_band)
    
    # Final dictionary containing scientific data   
    data_diz = {}
    
    # Iterate over each polarimeter
    for pol, dets in output_diz.items():
        data_diz[pol] = {}
            
        # Iterate over each detector
        for det, values in dets.items():
        
            time = values["time"]
            
            # Process scientific output
            if data_type == "DEM": 
                sci_data = double_dem(values["data"]) # Apply double_dem function if DEM data
            elif data_type == "PWR":
                sci_data = total_pwr(values["data"]) # Apply total_pwr function if PWR data
           
            # Save the results in the dictionary
            data_diz[pol][det] = {
                "time": average_time(time), # Timestamp in sec
                "data": sci_data, # Scientific data
            }
                
    return data_diz


def average_time(time):
    
    """
    Compute the average between each pair of consecutive time values.

    Args:
        time: 1D array of time values.

    Returns:
        avg_time: 1D array containing the average time values.
    """
    
    avg_time = (time[:-1:2] + time[1::2]) / 2
    
    return avg_time


def double_dem(dem):
    
    """
    Compute the difference between each pair of consecutive DEM data values, divided by 2.

    Args:
        dem: 1D array of DEM data values.

    Returns:
        diff: 1D array containing the double demodulated data values.
    """
    
    n = (len(dem) // 2) * 2
    diff = (dem[0:n:2] - dem[1:n:2])/2
    
    return diff


def total_pwr(pwr):
    
    """
    Compute the sum between each pair of consecutive PWR data values, divided by 2.

    Args:
        pwr: 1D array of PWR data values.

    Returns:
        avg: 1D array containing the total power data values.
    """
    
    n = (len(pwr) // 2) * 2
    avg = (pwr[0:n:2] + pwr[1:n:2])/2
    
    return avg

#--------------------------------------------------------------------------------------------------------------------------------
# Spectral analysis

def spectrum_data(data_diz, spectrum_type="PSD", remove_drift=True, welch=True, lowfreq=1e-3):
    
    """
    Compute the noise spectrum of the data, using the spectrum.Spectrum() class.

    Args:
        data_diz: Dictionary structured as {'polarimeter': {'detector': {'time', 'data'}}}.
        spectrum_type: Type of spectrum to compute ('AS', 'PS', 'ASD', 'PSD') (default value: 'PSD')
        remove_drift: If True remove a linear drift from data (default value: True)
        welch: If True apply Welch windowing with segments of length samp_freq/lowfreq (default value: True)
        lowfreq: Lowest frequency in the spectrum if welch=True (default value: 1e-3)

    Returns:
        spec_diz: Dictionary structured as {'polarimeter': {'detector': {'frequency', 'spectrum'}}}.
    """
    
    # Initialize the Spectrum class
    spec_inst = spectrum.Spectrum()
    spec_inst.spectrum_type[0] = spectrum_type # Calculate spectrum
    spec_inst.remove_drift[0] = remove_drift # If True remove a linear drift from data
    spec_inst.welch[0] = welch # If True apply Welch windowing with segments of length samp_freq/low_freq
    spec_inst.lowfreq[0] = lowfreq # Lowest frequency in the spectrum
    
    # Final dictionary containing spectrum data   
    spec_diz = {}
    
    # Iterate over each polarimeter
    for pol, dets in data_diz.items():
        spec_diz[pol] = {}
            
        # Iterate over each detector
        for det, values in dets.items():
        
            time = values["time"]
            data = values["data"]
                
            # Calculate sampling frequency
            T = np.mean(np.diff(time)) # Average time between two points
            samp_freq = 1 / T # Sampling frequency

            # Compute the noise spectrum
            spec = spec_inst.spectrum(data, samp_freq)
           
            # Save the results in the dictionary
            spec_diz[pol][det] = {
                "frequency": spec["frequencies"], # Spectrum frequency
                "spectrum": spec["amplitudes"], # Spectrum amplitude
            }
                
    return spec_diz

#--------------------------------------------------------------------------------------------------------------------------------
# Spike detection and removal

def find_spikes(spec_diz, save_path, window_points=50, threshold_sigma=4.5):
    
    """
    Identifies spikes in the PSD of TOD by detecting frequencies where the spectral amplitude significantly exceeds the local baseline.

    Args:
        spec_diz: Dictionary structured as {'polarimeter': {'detector': {'frequency', 'spectrum'}}}.
        save_path: Path where the detected spike frequencies will be saved as a JSON file.
        window_points: Number of points per frequency window used for local trend fitting (default value: 50).
        threshold_sigma: Threshold in units of the residual standard deviation to flag spikes (default value: 4.5).

    Returns:
        spike_freqs: Dictionary structured as {'polarimeter': {'detector': [spike frequencies]}}.
    """
    
    file_name = "spikes.json"
    save_file = os.path.join(save_path, file_name)
    
    #Final dictionary containing detected spike frequencies
    spike_freqs = {} 
    
    # Iterate over each polarimeter
    for pol, dets in spec_diz.items():
        spike_freqs[pol] = {}
        
        # Iterate over each detector
        for det, values in dets.items():
            frequencies = values["frequency"]
            amplitudes = values["spectrum"]
            spike_freqs[pol][det] = []
            
            # Iterate over the spectrum using non-overlapping windows
            for i in range(0, len(frequencies) - window_points, window_points):
                freq_win = frequencies[i:i + window_points]
                spec_win = amplitudes[i:i + window_points]
                
                # Convert to logarithmic scale to fit the spectral slope
                log_freq = np.log10(freq_win)
                log_spec = np.log10(spec_win)

                # Fit and remove the local spectral trend
                a, b = np.polyfit(log_freq, log_spec, 1)        
                drift = np.polyval([a, b], log_freq)            
                resid = log_spec - drift
                
                # Compute mean and standard deviation of residuals
                mean = np.mean(resid)
                std = np.std(resid)
                
                # Detect outliers above the sigma threshold
                for j in range(window_points):
                    if resid[j] > mean + threshold_sigma * std:
                        spike_freqs[pol][det].append(freq_win[j])

    with open(save_file, "w") as f:
        json.dump(spike_freqs, f, indent=2)
    
    print(f"JSON file saved to {save_file}")

    return spike_freqs


def remove_spikes(data_diz, window_sec):
    
    """
    Removes a periodic spike pattern from a TOD by computing and subtracting the average waveform over fixed-length windows.

    Args:
        data_diz: Dictionary structured as {'polarimeter': {'detector': {'time', 'data'}}}.
        window_sec: Duration of the window in seconds used for spike averaging.

    Returns:
        data_clean_diz: data_diz: Dictionary structured as {'polarimeter': {'detector': 'time', 'data (without periodic spikes)'}}}.
    """
    
    # Final dictionary containing spectrum data   
    data_clean_diz = {}                                 
    
    # Iterate over each polarimeter
    for pol, dets in data_diz.items():
        data_clean_diz[pol] = {}
            
        # Iterate over each detector
        for det, values in dets.items():
        
            time = values["time"]
            data = values["data"]            
                
            # Define window parameters
            dt = np.mean(np.diff(time)) # Average sampling time interval
            n_samples = int(np.round(window_sec / dt)) # Number of samples per window
            n_windows = int(len(data) / n_samples) # Number of windows

            # Compute the average spike pattern in a window
            spike = data[:n_windows * n_samples].reshape((n_windows, n_samples)) # Reshape data
            spike_avg = np.mean(spike, axis=0) # Average over all windows
            baseline = np.median(spike_avg) # Estimate baseline level
            spike_avg -= baseline # Remove baseline
            
            time_window = np.linspace(0, window_sec, n_samples, endpoint=False)
            
            # Initial guess
            A = np.max(spike_avg) - np.min(spike_avg) # Spike amplitude
            t0 = time_window[np.argmax(np.abs(spike_avg))]  # Time corresponding to the spike peak
            w = 0.1 * window_sec # Spike width (assumed as 10% of the window duration)
            guess = [A, t0, w]

            # Spike fitting
            popt, _ = curve_fit(
                lambda t, A, t0, w: square_smooth(t, A, t0, w),
                time_window,
                spike_avg,
                p0=guess
            )
            spike_fit = square_smooth(time_window, *popt)

            # Repeat the average spike pattern to match the signal length
            r = int(np.ceil(len(data) / n_samples))
            data_spikes = np.tile(spike_fit, r)[:len(data)]

            # Subtract the periodic component from the original signal
            data_cleaned = data - data_spikes
           
            # Save the results in the dictionary
            data_clean_diz[pol][det] = {
                "time": time, # Timestamp in sec
                "data": data_cleaned, # Scientific data with the periodic component removed
            }
                
    return data_clean_diz


def square_smooth(t, A, t0, w, k=100):
    
    """
    Generates a smoothed square pulse for modeling a spike waveform.

    Args:
        t: Array of time values.
        A: Spike amplitude.
        t0: Central time position of the spike.
        w: Spike duration.
        k: Smoothing parameter (default value: 100).

    Returns:
        spike: Array of vlaues representing the smoothed square pulse at each time t.
    """
    
    spike = A * 0.5 * (np.tanh(k*(t - (t0 - w/2))) - np.tanh(k*(t - (t0 + w/2))))
    
    return spike

#--------------------------------------------------------------------------------------------------------------------------------
# Save housekeeping and thermal data to Excel file

def save_housekeeping_data(ds, mjd_ranges, parameters, polarimeters, save_path, Q_band=True, W_band=True):
    
    """
    Save average housekeeping parameter values for each polarimeter and time range, into a single Excel file.

    Args: 
        ds: Database containing the data files.
        mjd_ranges: List of MJD time intervals [e.g. (start1, end1), (start2, end2), ...].
        parameters: Parameters to consider (e.g. ['VD0_HK', 'ID0_HK', 'VPIN0_HK', ...]).
        polarimeters: Polarimeters to consider (can be both a list of polarimeters e.g. ['R0', ...] and boards e.g. ['R', ...]).
        save_path: Directory where final data will be saved.
        Q_band: If True consider Q-band polarimeters (default value: True).
        W_band: If True consider W-band polarimeters (default value: True).
    """

    file_name = "housekeeping_data.xlsx"
    save_file = os.path.join(save_path, file_name)

    with pd.ExcelWriter(save_file) as writer:
        
        for mjd_range in mjd_ranges:
            sheet_name = Time(mjd_range[0], format="mjd").datetime.strftime("%Y_%m_%d_%H-%M-%S")
            row = {}

            data_diz = housekeeping_data(ds, mjd_range, parameters, polarimeters, Q_band, W_band)

            for pol in data_diz:
                for par in data_diz[pol]:
                    data = data_diz[pol][par]["data"]
                    row.setdefault(par, {})[pol] = round(np.nanmedian(data))

            df = pd.DataFrame(row).T
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name)

    print(f"Excel file saved to {save_file}")

    
def save_thermal_data(ds, mjd_ranges, sensor_name, save_path, get_raw=False):
    
    """
    Save average temperature values for each thermal sensor and time range, into a single Excel file.

    Args: 
        ds: Database containing the data files.
        mjd_ranges: List of MJD time intervals [e.g. (start1, end1), (start2, end2), ...].
        sensor_name: Thermal sensors to consider (e.g. ['TS-CX1-Module-R', 'EX-CX18-SpareCx', ...]).
        save_path: Directory where final data will be saved.
        get_raw: If True get raw values [ADU], if False get calibrated values [K] (default value: False).
    """

    file_name = "thermal_data.xlsx"
    save_file = os.path.join(save_path, file_name)
    
    title = "Temperature [ADU]" if get_raw else "Temperature [K]"
    title_err = "ΔT [ADU]" if get_raw else "ΔT [K]"

    with pd.ExcelWriter(save_file) as writer:
        
        for mjd_range in mjd_ranges:
            sheet_name = Time(mjd_range[0], format="mjd").datetime.strftime("%Y_%m_%d_%H-%M-%S")
            row = {}

            data_diz = thermal_data(ds, mjd_range, sensor_name, get_raw)

            for sensor in data_diz:
                data = data_diz[sensor]["data"]
                mean_val = round(np.nanmean(data), 3)        
                err_val = round((np.nanmax(data) - np.nanmin(data)) / 2, 3)
                row.setdefault(sensor, {})[title] = mean_val
                row[sensor][title_err] = err_val

            df = pd.DataFrame(row).T
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name)

    print(f"Excel file saved to {save_file}")

#--------------------------------------------------------------------------------------------------------------------------------
# Plot

def save_plot(fig, mjd_range, save_path, dir_name, plot_name):
    
    file_date = Time(mjd_range[0], format="mjd").datetime.strftime("%Y_%m_%d_%H-%M-%S")
    save_dir = os.path.join(save_path, file_date, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    plot_name += ".png"
    save_fig = os.path.join(save_dir, plot_name)

    fig.savefig(save_fig)
    
    plt.close(fig)
    

def plot_data(data_diz, mjd_range, save_path=None, save=False):
    biases = InstrumentBiases()
    t0 = Time(mjd_range[0], format="mjd").unix

    for pol, pars in data_diz.items():
        pol_name = biases.module_name_to_polarimeter(f"{pol}")
            
        for par, values in pars.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            time = values["time"] - t0
            data = values["data"]
            ax.plot(time, data)
            ax.set_title(f"POL_{pol} ({pol_name}) - {par}", fontsize=24)
            ax.set_xlabel("Time [s]", fontsize=18)
            ax.set_ylabel("Data [ADU]", fontsize=18)
            ax.grid(True, which="both", linestyle="--")
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            fig.tight_layout()

            if save:
                save_plot(fig, mjd_range, save_path, f"{pol}", f"{pol}_{par}") 
                
                
def plot_spec(spec_diz, mjd_range, save_path=None, save=False):
    biases = InstrumentBiases()
    
    for pol, pars in spec_diz.items():
        pol_name = biases.module_name_to_polarimeter(f"{pol}")
            
        for par, values in pars.items():
            fig, ax = plt.subplots(figsize=(6, 5))
            frequencies = values["frequency"]
            amplitudes = values["spectrum"]
            ax.loglog(frequencies, amplitudes)
            ax.set_title(f"POL_{pol} ({pol_name}) - {par}", fontsize=24)
            ax.set_xlabel("Frequency [Hz]", fontsize=18)
            ax.set_ylabel(r"PSD [ADU$^2$/Hz]", fontsize=18)
            ax.grid(True, which="both", linestyle="--")
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=10)
            fig.tight_layout()

            if save:
                save_plot(fig, mjd_range, save_path, f"{pol}", f"{pol}_{par}")  