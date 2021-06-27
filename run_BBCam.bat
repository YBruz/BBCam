rem make sure to install all dependencies in requirements.txt before running BBCam
rem this batch file will only work if python is recognised by the environment
rem 
rem available arguments are:
rem --fps         sets the fps of the webcam         default: 20
rem --mtresh      sets the bodypix mask treshhold    default: 0.75
rem --mdil        sets the mask dilation value       default: 51
rem --msmooth     sets the mask smoothing value      default: mdil * 1.5
rem --mblur       sets the mask feathering value     default: 51
rem --bblur       sets the background blur value     default: 31
rem --headless    hides the webcam window            default: false
rem

python BBCam.py --bblur=51
pause