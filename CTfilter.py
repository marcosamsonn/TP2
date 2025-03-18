#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# libs
import numpy as np

## filtrer le sinogramme
## ligne par ligne
def filterSinogram(sinogram):
    for i in range(sinogram.shape[0]):
        sinogram[i] = filterLine(sinogram[i])

## filter une ligne (projection) via FFT
def filterLine(projection):
    fft_proj = np.fft.fft(projection)

    # Taille du filtre
    n = len(projection)

    # Fréquences
    # fftshift pour avoir les fréquences dans l'ordre [-N/2, ..., -1, 0, 1, ..., N/2-1]
    freq = np.fft.fftshift(np.fft.fftfreq(n))

    # Filtre rampe
    ramp_filter = np.abs(freq)

    # Appliquer le filtre
    proj_shifted = np.fft.fftshift(fft_proj)
    filtered_proj = proj_shifted * ramp_filter

    # Inverser le shift
    filtered_proj = np.fft.ifftshift(filtered_proj)

    # Inverser la FFT
    filtered_proj = np.fft.ifft(filtered_proj)

    return np.real(filtered_proj)

