# Welcome to ESSC2030 - Lecture11 Application of Geoscience 
## Tutorial on Stereonegraphic Projection and Stereonet on Python

This tutorial will introduce using Python to analysis and illustrate 3D
geological structures using Stereonet. First of all, there are three suggested methods to
conduct the tutorial. 

1. Pre-build Docker - Binder [Suggested]
    
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jwjeremy/ESSC2030_lec11/master)
        
    The link will connect to jupyter notebook of the repository that can run remotely.
        
2. Google Colab 

    Open a webpage and go to https://colab.research.google.com/

    Select Github on the fourth coloumn in the colored section

    Paste the following link as GitHub URL
        https://github.com/jwjeremy/ESSC2030_lec11

    Select lec11_stereo/Lec11_tut_stereo [Student_version].ipynb
        or lec11_stereo/Lec11_tut_stereo.ipynb if you want answer

    Please remember to save the session to your own google drive to leave
    records on your progress.

    Please also upload the data file required in the tutorial to the google
    colab, which are lai_chi_chong_fold.data and Lai_chi_chong_bedding.data
    respectively.

    The use of google colab is same as jupyter notebook. It utilizes google server
    to support the ipython console at backend, which allows us to run on a webpage.

    On the left column, you can select table of content to navigate throughout
    the tutorial. 

    Please be noted that it is unable to save the record automatically. You can
    download the .ipynb and replace the one in the tutorial package.
    

3. Run locally

    Since the following tutorial requires python2 and relevant packages, please
    create a new conda environment using the lec11_stereo.yml

    conda env create -f lec11_stereo.yml

    To activate the installed environment, please type
        activate conda activate lec11_stereo

    and start the tutoral by creating jupyter notebook session
        jupyter notebook


Other intstructions are written in the .ipynb. Have fun with the tutorial~


The tutorial package includes the following files

    ├── Lai_chi_chong_bedding.data
    ├── Lec11_tut_stereo\ [Student_version].ipynb
    ├── Lec11_tut_stereo.ipynb
    ├── README
    ├── fig
    │   ├── Fold_axis_method.gif
    │   ├── LCC_slumpfoldaxe.png
    │   ├── LLC_slumpfold.jpg
    │   └── Stereographic_projection.jpg
    ├── lai_chi_chong_fold.data
    └── lec11_stereo.yml

Latest updates of the tutorial can be found in: https://github.com/jwjeremy/ESSC2030_lec11


Latest update: 2020/09 by JW 
