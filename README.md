# Camera Tampering Detection

Opencv based Python implementation of  real time robust camera tampering detection algorithms :

## Moving of the Camera 
    
Background and foreground segmentation is done using Gaussian mixture models .Thresholding of the foreground area is done to determine whether the camera has been moved or not .

### reference : https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html


## Occlusion of the Camera 

Moving average of the Variance of the image is calculated for the past 20 and 100 frames respectivley , if the difference b/w them falls below the specified threshold the video is flagged for tampering .


   
## Defocusing of the Camera 

Sharpness of the image is calculated using Laplacian filter , Moving average of the variance of the image obtained after laplacian operation is calculated for the past 20 and 100 frames respectivley , if the difference b/w them falls below a certain threshold the video is flagged for tampering .

### reference : https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html

## Temproal similarity between frames 

SSIM image similarity metric is calculated between the frame and segmented background at every 100 steps , if the similarity metric falls below the specified threshold the video is flagged for tampering . 

### reference : https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html

## Running the script :
    

Default values of variables : 

video_path = 0 

output_path = output.avi

area_thresh = 0.33

occ_thresh = 0.5

blur_thresh = 0.5

ssim_thresh = 0.6 

### To run with default values :

python tampering detection.py

### To run with specified values values :

export variable_name = value

python tampering_detection.py --video_path ${video_path} --output_path ${output_path} --area_thresh ${area_thresh} --occ_thresh ${occ_thresh} --blur_thresh ${blur_thresh} --ssim_thresh ${ssim_thresh}

