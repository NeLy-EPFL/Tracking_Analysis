# Tracking_Analysis
This repository contains processing, analysis and plotting scripts used for the research projects of Matthias Durrieu. The repository is organised as follows:

## Ball Pushing
Object affordance project using the Optobot and the Multimaze recorder to explore flies ability to push balls. In each subdirectory, associated processing and analysis scripts can be found.

## Food Objects

Another object affordance project using edible agar-based objects to train flies to recognise physical multisensory objects with learnt values in their environment. 
In each subdirectory, associated processing and analysis scripts can be found.

## Tracktor

A tracking algorithm published by Vivek Sridhar and publicly available here: https://github.com/vivekhsridhar/tracktor

Here only the core tracktor module was kept from the original project and was slightly modified to fit the needs of the projects. 

Tracktor is mainly used in the Food Objects project but the [fly tracker](Food_Objects/Tracking/TrackFly.py) implemented there can be adapted to many other projects.

## Utilities

This directory contains a few useful scripts that can be used in many projects. It includes images and video manipulation tools, as well as some processing (filtering etc.) and analysis (Bootstrap, etc.) scripts. [Processing](Utilities/Processing.py) and [Utils](Utilities/Utils.py) are often imported in other scripts in this repository, as they are basically library of functions, whereas the other scripts are intended to be run as standalone scripts.