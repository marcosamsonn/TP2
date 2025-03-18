#!/usr/bin/env python
# -*- coding: utf-8 -*-
# TP reconstruction TDM (CT)
# Prof: Philippe Després
# programme: Dmitri Matenine (dmitri.matenine.1@ulaval.ca)


# libs
import numpy as np
import time

# local files
import geo as geo
import util as util
import CTfilter as CTfilter

## créer l'ensemble de données d'entrée à partir des fichiers
def readInput():
    # lire les angles
    [nbprj, angles] = util.readAngles(geo.dataDir+geo.anglesFile)

    print("nbprj:",nbprj)
    print("angles min and max (rad):")
    print("["+str(np.min(angles))+", "+str(np.max(angles))+"]")

    # lire le sinogramme
    [nbprj2, nbpix2, sinogram] = util.readSinogram(geo.dataDir+geo.sinogramFile)

    if nbprj != nbprj2:
        print("angles file and sinogram file conflict, aborting!")
        exit(0)

    if geo.nbpix != nbpix2:
        print("geo description and sinogram file conflict, aborting!")
        exit(0)

    return [nbprj, angles, sinogram]


## reconstruire une image TDM en mode rétroprojection
def laminogram():
    
    [nbprj, angles, sinogram] = readInput()

    # initialiser une image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))

    # Coordonnées du centre
    center_x = geo.nbvox/2
    center_y = geo.nbvox/2

    # "etaler" les projections sur l'image
    # ceci sera fait de façon "voxel-driven"
    # pour chaque voxel, trouver la contribution du signal reçu
    for j in range(geo.nbvox): # colonnes de l'image, nbvox = taille du détecteur = 336
        print("working on image column: "+str(j+1)+"/"+str(geo.nbvox))
        for i in range(geo.nbvox): # lignes de l'image
            for a in range(len(angles)):
                # Pour chaque voxel, calcul des coordonnées cartésiennes par rapport au centre de l'image
                # Le centre de l'image est à l'origine.
                x = (center_x - j) * geo.voxsize
                y = (i - center_y) * geo.voxsize

                # Pour un angle donné, on projette le point (x,y) sur la droite du détecteur
                # t est la position relative sur le détecteur.
                t = x * np.cos(angles[a]) + y * np.sin(angles[a])

                # Convertir t en indice de pixel sur détecteur
                # C'est un float de la position exact où le rayon touche le détecteur
                detector_idx = geo.nbpix/2 + t/geo.pixsize

                # On doit donc interpoler pour obtenir la valeur du sinogramme à cet indice
                # Récupérer la valeur
                # Vérif si indice dans range du détecteur.
                if 0 <= detector_idx < geo.nbpix:
                    # Interpolation linéaire
                    idx_floor = int(detector_idx)  # donne uniquement la partie entière (round down)
                    idx_ceil = min(idx_floor + 1, geo.nbpix-1)  # donne le pixel suivant
                    weight = detector_idx - idx_floor
                    if 0 <= idx_floor < geo.nbpix:
                        valeur_interpolee = (1-weight) * sinogram[a, idx_floor] + weight * sinogram[a, idx_ceil]

                        # Ajoute la valeur au voxel
                        image[i, j] += valeur_interpolee


    util.saveImage(image, "lam")


def laminogram_optimized():

    [nbprj, angles, sinogram] = readInput()

    # Initialiser une image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))

    # Coordonnées du centre
    center_x = geo.nbvox / 2
    center_y = geo.nbvox / 2

    # Matrices 2D des coordonnées x et y de chaque voxel
    x_indices, y_indices = np.meshgrid(np.arange(geo.nbvox), np.arange(geo.nbvox))
    x_coords = (center_x - x_indices) * geo.voxsize
    y_coords = (y_indices - center_y) * geo.voxsize

    # Pour chaque angle
    for a in range(len(angles)):
        print(f"Processing angle {a + 1}/{len(angles)}")

        # Calculer la position sur le détecteur pour tous les voxels d'un coup
        t = x_coords * np.cos(angles[a]) + y_coords * np.sin(angles[a])

        # Convertir t en indices sur le détecteur
        detector_indices = geo.nbpix/2 + t/geo.pixsize

        # Créer un masque pour les indices valides
        valid_indices = (detector_indices >= 0) & (detector_indices < geo.nbpix - 1)

        # Pour l'interpolation linéaire
        indices_floor = np.floor(detector_indices).astype(int)
        weights = detector_indices - indices_floor

        # Interpolation linéaire sur les indices valides
        valid_floor = np.clip(indices_floor[valid_indices], 0, geo.nbpix - 2)
        valid_ceil = valid_floor + 1

        # Matrice temporaire pour les valeurs de cet angle
        temp_image = np.zeros_like(image)

        # Calculer les valeurs interpolées
        floor_values = sinogram[a, valid_floor]
        ceil_values = sinogram[a, valid_ceil]
        interpolated = (1 - weights[valid_indices]) * floor_values + weights[
            valid_indices] * ceil_values

        # Mettre valeurs aux indices
        temp_image[valid_indices] = interpolated

        # Ajouter à l'image
        image += temp_image

    util.saveImage(image, "lam_fast")


## reconstruire une image TDM en mode retroprojection filtrée
def backproject():
    
    [nbprj, angles, sinogram] = readInput()
    
    # initialiser une image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))
    
    ### option filtrer ###
    CTfilter.filterSinogram(sinogram)
    ######

    # Coordonnées du centre
    center_x = geo.nbvox / 2
    center_y = geo.nbvox / 2

    # Matrices 2D des coordonnées x et y de chaque voxel
    x_indices, y_indices = np.meshgrid(np.arange(geo.nbvox), np.arange(geo.nbvox))
    x_coords = (center_x - x_indices) * geo.voxsize
    y_coords = (y_indices - center_y) * geo.voxsize

    # Pour chaque angle
    for a in range(len(angles)):
        print(f"Processing angle {a + 1}/{len(angles)}")

        # Calculer la position sur le détecteur pour tous les voxels d'un coup
        t = x_coords * np.cos(angles[a]) + y_coords * np.sin(angles[a])

        # Convertir t en indices sur le détecteur
        detector_indices = geo.nbpix / 2 + t / geo.pixsize

        # Créer un masque pour les indices valides
        valid_indices = (detector_indices >= 0) & (detector_indices < geo.nbpix - 1)

        # Pour l'interpolation linéaire
        indices_floor = np.floor(detector_indices).astype(int)
        weights = detector_indices - indices_floor

        # Interpolation linéaire sur les indices valides
        valid_floor = np.clip(indices_floor[valid_indices], 0, geo.nbpix - 2)
        valid_ceil = valid_floor + 1

        # Matrice temporaire pour les valeurs de cet angle
        temp_image = np.zeros_like(image)

        # Calculer les valeurs interpolées
        floor_values = sinogram[a, valid_floor]
        ceil_values = sinogram[a, valid_ceil]
        interpolated = (1 - weights[valid_indices]) * floor_values + weights[
            valid_indices] * ceil_values

        # Mettre valeurs aux indices
        temp_image[valid_indices] = interpolated

        # Ajouter à l'image
        image += temp_image
    
    util.saveImage(image, "fbp")


## reconstruire une image TDM en mode retroprojection
def reconFourierSlice():
    
    [nbprj, angles, sinogram] = readInput()

    # initialiser une image reconstruite, complexe
    # pour qu'elle puisse contenir sa version FFT d'abord
    IMAGE = np.zeros((geo.nbvox, geo.nbvox), 'complex')
    
    # conteneur pour la FFT du sinogramme
    SINOGRAM = np.zeros(sinogram.shape, 'complex')

    #image reconstruite
    image = np.zeros((geo.nbvox, geo.nbvox))
    #votre code ici
   #ici le défi est de remplir l'IMAGE avec des TF des projections (1D)
   #au bon angle.
   #La grille de recon est cartésienne mais le remplissage est cylindrique,
   #ce qui fait qu'il y aura un bon échantillonnage de IMAGE
   #au centre et moins bon en périphérie. Un TF inverse de IMAGE vous
   #donnera l'image recherchée.

   
    
    util.saveImage(image, "fft")


## main ##
start_time = time.time()
#laminogram()
#laminogram_optimized()
backproject()
#reconFourierSlice()
print("--- %s seconds ---" % (time.time() - start_time))




# Réponse Question 2 : Z8HA9
"""
La rétroprojection simple créera une image floue. Chaque projection étale ses valeurs 
sur des lignes entières, ce qui crée des "trainées" dans l'image reconstruite. C'est
pourquoi les méthodes plus avancées comme la rétroprojection filtrée (FBP) sont
utilisées en pratique
"""