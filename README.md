# HOG-SVM-to-detect-head
## This project uses HOG(Histogram of Oriented Gradient) and SVM(Support Vector Machine) to do head detection. </br>
### Package details:
Python 3.6 </br>
scikit-image        0.13.1   </br>  
scikit-learn        0.19.1   </br>
numpy               1.12.1   </br>

### Code details: </br>
- nms.py:  
    This function performs Non-Maxima Suppression. </br>
    `detections` consists of a list of detections. </br>
    Each detection is in the format -> </br>
    [x-top-left, y-top-left, confidence-of-detections, width-of-detection, height-of-detection] </br>
    If the area of overlap is greater than the `threshold`, </br>
    the area with the lower confidence score is removed. </br>
    The output is a list of detections. </br>
- extractFeat.py
    This function performs extracting feature maps woth HOG and storing those feature data.
- classifier.py 
