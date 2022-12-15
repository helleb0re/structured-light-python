# **Structured light project**

This project is an attempt to create an easy-to-understand and flexible-to-use framework for implementing the Fringe Projection Profilometry (FPP) method in the Python.

## **Implementation**

* Ability to use any camera (modules with webcams through OpenCV and Baumer cameras through NeoAPI are implemented in the project)
* The Phaseshift Projection Profilometry (PSP) method with sinusoidal fringes is implemented to obtain phase fields
* Projection pattern generation supports an arbitrary number of phase shifts and an arbitrary number of periods 
* A hierarchical approach is used to unwrap the phase fields
* Implemented automatic detection of the fringe projection area on the images (ROI)
* A simple gamma correction method for projected images is implemented
* Flexible adjustment of the experiment and hardware parameters with the help of config files

## **How to use**

1. Install depedicies
```
pip install opencv-contrib-python numpy scipy matplotlib
```
2. Setting the parameters of the experiment and the hardware in the file `config.py`

3. Launch main module
```
python main.py
```

In the script `examples/test_plate_phasogrammetry.py` there is an example of processing the results of the experiment to determine the shape of the surface of a granite slab using the phasogrammetric approach. To date, the measurement accuracy of about **60 µm** has been achieved.

## **References**
The following sources were used to implement the algorithms

[Zuo C. et al. Phase shifting algorithms for fringe projection profilometry: A review // Optics and Lasers in Engineering. 2018. Vol. 109. P. 23-59.](https://doi.org/10.1016/j.optlaseng.2018.04.019)

[Zuo C. et al. Temporal phase unwrapping algorithms for fringe projectionprofilometry: A comparative review // Optics and Lasers in Engineering. 2016. Vol. 85. P. 84-103.](https://doi.org/10.1016/j.optlaseng.2016.04.022)

[Feng S. et al. Calibration of fringe projection profilometry: A comparative review // Optics and Lasers in Engineering. 2021. Vol. 143. P. 106622.](https://doi.org/10.1016/j.optlaseng.2021.106622)

[Zhong K. et al. Pre-calibration-free 3D shape measurement method based on fringe projection // Optics Express. 2016. Vol. 24. №. 13. P. 14196-14207.](https://doi.org/10.1364/OE.24.014196)

## **Authors**
Anton Poroykov, Ph.D., associated professor 

Nikita Sivov, graduate student 

## **Acknowledgements**
The research was carried out at the expense of the grant Russian Science Foundation No. 22-21-00550 (https://rscf.ru/project/22-21-00550/).