#!/usr/bin/env python

# Print out the brightest stars that should be visible withing 30
# degrees of the zenith in a given all sky camera image.

# Usage:
# ./asc_find_visible_stars.py asc_image.fits
#
# Based on programs written by Jamie Paltridge and Queenette Anandi
# during their final year project with the Astrophysics group at
# the University of Bristol


from astropy import units as u
from astropy.coordinates import SkyCoord, Angle, EarthLocation, AltAz
import math
from astropy.utils.iers import conf
from astropy.time import Time
import sys
from astropy.io import fits
import numpy as np
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from astropy.stats import mad_std

# Altitude limit is set here. eg if alt_limit = 50, no stars with
# altitudes below 50 will be considered
alt_limit=50.0

# How many stars to output
num_stars=10

imagefile = sys.argv[1]
print("Opening FITS file: ", imagefile)

# Get image header
hdu = fits.open(imagefile)[0]

# Get datestamp from FITS header
tstring = hdu.header['DATE']

print("Date and time of FITS file is ",tstring)

# Need this to stop IERS errors
conf.auto_max_age = None

# Set time and date here
timestr = tstring

# Choose your star names below as a list

star_names = [
"Sirius", "Canopus", "Arcturus", "Vega", "Capella", "Rigel",
"Procyon", "Achernar", "Betelgeuse", "Hadar", "Altair", "Acrux",
"Aldebaran", "Antares", "Spica", "Pollux", "Fomalhaut", "Deneb",
"Mimosa", "Regulus", "Adhara", "Shaula", "Castor", "Gacrux",
"Bellatrix", "Elnath", "Miaplacidus", "Alnilam", "Gamma Velorum",
"Alnair", "Alnitak", "Alioth", "Dubhe", "Mirfak", "Wezen", "Sargas",
"Kaus Australis", "Avior", "Alkaid", "Menkalinan", "Atria", "Alhena",
"Peacock", "Alsephina", "Mirzam", "Alphard", "Polaris", "Hamal",
"Algieba", "Diphda", "Mizar", "Nunki", "Menkent", "Mirach",
"Alpheratz", "Rasalhague", "Kochab", "Saiph", "Denebola", "Algol",
"Tiaki", "Aspidiske", "Suhail", "Alphecca", "Mintaka", "Sadr",
"Eltanin", "Schedar", "Naos", "Almach", "Caph", "Izar", "Dschubba",
"Larawag", "Merak", "Ankaa", "Enif", "Scheat", "Sabik", "Phecda",
"Aludra", "Markeb", "Markab", "Aljanah", "Acrab"]


# Magnitude values corresponding to each star

star_magnitudes = {
    "Sirius": -1.46, "Canopus": -0.74, "Arcturus": -0.05, 
    "Vega": 0.03, "Capella": 0.08, "Rigel": 0.13, "Procyon": 0.34, 
    "Achernar": 0.46, "Betelgeuse": 0.50, "Hadar": 0.61, 
    "Altair": 0.76, "Acrux": 0.76, "Aldebaran": 0.86,
    "Antares": 0.96, "Spica": 0.97, "Pollux": 1.14, "Fomalhaut": 1.16,
    "Deneb": 1.25, "Mimosa": 1.25, "Regulus": 1.39, "Adhara": 1.50,
    "Shaula": 1.62, "Castor": 1.62, "Gacrux": 1.64, "Bellatrix": 1.64,
    "Elnath": 1.65, "Miaplacidus": 1.69, "Alnilam": 1.69, "Gamma Velorum": 1.72, 
    "Alnair": 1.74, "Alnitak": 1.77, "Alioth": 1.77,
    "Dubhe": 1.79, "Mirfak": 1.80, "Wezen": 1.82, "Sargas": 1.84,
    "Kaus Australis": 1.85, "Avior": 1.86, "Alkaid": 1.86,
    "Menkalinan": 1.90, "Atria": 1.91, "Alhena": 1.92, "Peacock":
    1.94, "Alsephina": 1.96, "Mirzam": 1.98, "Alphard": 2.0,
    "Polaris": 1.98, "Hamal": 2.0, "Algieba": 2.08, "Diphda": 2.02,
    "Mizar": 2.04, "Nunki": 2.05, "Menkent": 2.06, "Mirach": 2.06,
    "Alpheratz": 2.07, "Rasalhague": 2.08, "Kochab": 2.09, "Saiph":
    2.11, "Denebola": 2.12, "Algol": 2.15, "Tiaki": 2.17, "Aspidiske":
    2.21, "Suhail": 2.23, "Alphecca": 2.23, "Mintaka": 2.23, "Sadr":
    2.23, "Eltanin": 2.24, "Schedar": 2.25, "Naos": 2.26, "Almach":
    2.28, "Caph": 2.29, "Izar": 2.31, "Dschubba": 2.31, "Larawag":
    2.37, "Merak": 2.38, "Ankaa": 2.40, "Enif": 2.42, "Scheat": 2.43,
    "Sabik": 2.44, "Phecda": 2.45, "Aludra": 2.46, "Markeb": 2.48,
    "Markab": 2.48, "Aljanah": 2.5, "Acrab": 2.5
    }

# Latitude of Bristol is 51 deg 26' N, 02 deg 35' W
lat_degrees, lon_degrees = 51.4588, -2.6021

# Create Time object
time_obj = Time(timestr, format='isot', scale='utc')

# List to store star names with altitude greater than alt_limit
high_altitude_stars = []

# Iterate over each star name in the list
for starname in star_names:

    # Create coordinate object
    star_coord = SkyCoord.from_name(starname, frame='icrs')

    # convert RA and dec strings to degrees using astropy.coordinates
    ra_degrees, dec_degrees = star_coord.ra.degree, star_coord.dec.degree

    # Calculate Sidereal time for Bristol at the specified time
    sidereal_time = time_obj.sidereal_time('apparent', lon_degrees)

    sidereal_time_angle = Angle(sidereal_time, u.hourangle)

    hour_angle = sidereal_time.degree - ra_degrees

    hour_angle += 360.0 if hour_angle < 0 else 0

    sin_alt = (math.sin(math.radians(dec_degrees)) * math.sin(math.radians(lat_degrees))
               + math.cos(math.radians(dec_degrees)) * math.cos(math.radians(lat_degrees))
               * math.cos(math.radians(hour_angle)))


    altitude = math.degrees(math.asin(sin_alt))

    bristol_location = EarthLocation(lat=lat_degrees*u.deg, lon=lon_degrees*u.deg)
    
    staraltaz = star_coord.transform_to(AltAz(obstime=time_obj, location=bristol_location))

    # Check if altitude is greater than 60 and add the star name and 
    # magnitude to the list
    if staraltaz.alt.degree > alt_limit:
        high_altitude_stars.append((starname, star_magnitudes[starname]))

# Sort the list based on magnitude values
high_altitude_stars.sort(key=lambda x: x[1])

# Extract only the star names from the sorted list
star_names = [star[0] for star in high_altitude_stars]

sorted_star_names = star_names[:num_stars]

# Print the list of star names with altitude greater than 60, ordered by magnitude

print("Brightest stars with altitude greater than 60, ordered by magnitude, at the date and time of the FITS image:", sorted_star_names)

# Open image

#image_data = fits.getdata(imagefile)

# Now open mask file
#maskfile = 'mask_final.fits'

#mask_data = fits.getdata(maskfile)

# Multiply the image by the mask
#sky_only_image = np.multiply(image_data, mask_data)

#mean, median, std = sigma_clipped_stats(sky_only_image, sigma=3.0)

#bkg_sigma = mad_std( sky_only_image )


#daofind = DAOStarFinder(fwhm=4., threshold=3.*bkg_sigma)

# Run the DAO starfinder software

#sources = daofind(sky_only_image)


#Sort data by magnitude

#sources.sort('mag')


#20 Brightest stars

#bright_stars = sources[:20]


#bright_stars = sources[:len(sorted_star_names)]

#bright_stars['id'] = sorted_star_names


#print("The x-y corrdinates for the brightest stars in the image are:")
#for row in bright_stars:
#    print(row[1], row[2])

