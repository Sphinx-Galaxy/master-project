# General Information

This project discusses the use of A.I in satellite missions for telemetry analysis and anomaly detection.

In the respectives subfolders are:
- 'Scripts' Various python scripts for an easier workflow
- 'Tensorflow' All A.I scripts, models, testdata and results
- 'Documentation' The project documentation in Latex

# Project structure

The working title of project is "Examination of A.I Machine Learning Algorithms and Definition of a Framework as Service for Anomaly Detection in Spacecraft On-Board Data".

## State of the Art - Introduction

The introduction covers the definition of an anomaly to try to qualify and quantify it. Second, the current state of developement is presented, with the project ATHMoS as main focus. This includes a discussion about feasable technqiues.

## Anomaly Detection Techniques

Here the discussed techniques are setup and tested with artifical datasets. Their results are briefly discussed regarding detection rate and programming complexity.

## Framework

The framework refers mostly to the Model Management Service developed at DLR.

## Implementation

Building of models in python and saving them as serial-buffers. This allows us to read and execute them in a C program. The C code is also shown in detail an explained.

## Validation

Check if the models work as intended in the lab as well as on the payload device.
