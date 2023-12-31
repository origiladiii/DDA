
# 1. example_save_to_file.py

## Overview
This Python script is designed for detecting and removing duplicate images and videos in a given directory. It utilizes various image processing techniques and computer vision libraries to analyze media files and identify duplicates based on perceptual hashing, color histograms, mean squared error, and video duration.

## Key Features
Perceptual Hashing: Identifies similar images using perceptual hashing (pHash).
Color Histogram Analysis: Compares images based on color distribution.
Mean Squared Error (MSE) Calculation: Assesses similarity by computing MSE.
Duplicate Video Detection: Analyzes video files for duplicate content.
Progress Tracking: Provides a console-based progress bar for long-running processes.
## Dependencies
os
cv2 (OpenCV)
PIL (Python Imaging Library)
imagehash
numpy
multiprocessing
time
json
sys
## Installation
Ensure Python is installed on your system.
Install required libraries: pip install opencv-python pillow imagehash numpy.


# Extended Explanation and Flow of the Media Duplication Detection and Removal Script
## Overview
The script is designed to detect and remove duplicate media files (both images and videos) within a specified directory. It employs a series of image processing techniques and algorithms to analyze and compare media files, leveraging perceptual hashing, color histograms, mean squared error (MSE), and video duration to identify duplicates.

# Flow of the Script
## 1. Initialization:

When executed, the script first sets the media_data_file path and calls the process_directory function with the specified directory path, the path of the media data file, the number of processing cores, and a flag to calculate MSE or not.
## 2. Directory Processing (process_directory):

It starts by gathering all media files (images and videos) from the specified directory.
For each media file, the script computes and stores its data, including perceptual hash, color histogram, and optionally MSE, for later comparison.
## 3. Media File Comparison and Duplicate Removal (remove_duplicates):

The script generates all possible pairs of media files for comparison.
Each pair is processed in parallel (using multiple cores) to identify duplicates.
For each pair, the script checks if they are duplicates based on the stored data (perceptual hash, histogram, and MSE).
If duplicates are found, the script removes the lower-quality file or the shorter video.
## 4. Detailed Functionality:

#### Perceptual Hashing (compute_phash):
Computes a 'perceptual hash' for an image, which is a fingerprint of the visual content, useful for comparing similarity.
#### Histogram Generation (get_histogram):
Creates a color histogram for an image, representing the distribution of colors in the image.
#### Histogram Comparison (compare_histograms): 
Compares two histograms to determine how similar two images are in terms of color distribution.
#### MSE Calculation (calculate_mse): 
Calculates the mean squared error between two images, a measure of similarity in terms of pixel-by-pixel comparison.
#### Frame Matching (is_matching_frame): 
Determines if two frames (from videos or images) are similar based on the above metrics.
#### Video Processing (get_video_frames, are_videos_duplicates): 
Extracts frames from videos and compares them to identify duplicate videos.
Video Duration Calculation (get_video_duration): Determines the duration of a video, used to compare video lengths as a preliminary duplicate check.
#### Quality Scoring (get_image_quality_score): 
Assigns a quality score to an image based on its resolution.
#### Parallel Processing:

The script utilizes Python's multiprocessing to process multiple media file pairs simultaneously, speeding up the duplicate detection process.
### Final Cleanup:
After processing, the script updates the media data file with the remaining media files.
A progress bar is displayed throughout the process to indicate the progress of the script.
# Conclusion
This script is a comprehensive solution for identifying and removing duplicate media files in a given directory. It efficiently processes large sets of media files by leveraging advanced image processing techniques and parallel computing, ensuring a thorough and speedy execution. The usage of perceptual hashing, histogram comparison, and MSE calculations provides a robust approach to identifying duplicates, even when the files are not identical but visually similar.

.
.
.

# 2. example_no_save_to_file.py

## Overview
This Python script is specifically designed for identifying and removing duplicate images and videos from a specified directory. It employs key image processing techniques like perceptual hashing, color histogram comparison, and mean squared error calculation to detect duplicates among a large set of media files.

# How It Works
## Initial Setup
When executed, the script starts by calling the process_directory function with the specified directory path.
It uses gather_all_media_files to find all media files (images and videos) in the specified directory.
## Core Functionality
### 1. Media File Analysis:

For each media file, the script performs different analyses based on the file type (image or video).
Images are analyzed using perceptual hashing and color histogram techniques.
Videos are processed frame by frame, applying similar image analysis techniques to each frame.
### 2. Duplicate Detection:

The script compares each media file with others in the directory.
For videos, it compares frames extracted from different videos for duplication.
For images, it compares perceptual hashes and histograms.
### 3. Duplicate Removal:

Once duplicates are identified, the script removes the lower-quality file or the shorter video.
This is done by comparing image resolutions and video durations.
## Detailed Functions
#### compute_phash(image, hash_size=8): 
Computes a perceptual hash for an image.
#### get_histogram(image): 
Generates a color histogram for an image.
#### compare_histograms(hist1, hist2): 
Compares two histograms.
#### is_matching_frame(ref_frame, check_frame, ...): 
Determines if two frames are similar.
#### get_video_frames(video_file, fps=16): 
Extracts frames from a video file.
#### are_videos_duplicates(ref_video, check_video, ...): 
Checks if two videos are duplicates.
#### get_video_duration(video_file): 
Gets the duration of a video.
#### get_image_quality_score(image_path): 
Calculates a quality score for an image based on its resolution.
#### print_progress(iteration, total, ...): 
Prints a progress bar in the console.
#### remove_duplicates(media_files, ...): 
Main function that processes and removes duplicates.
#### is_valid_image(input): 
Validates if the input is an image.
#### gather_all_media_files(directory_path): 
Collects all media files in a directory.
#### process_directory(main_directory_path): 
Initiates the processing of the directory for duplicate removal.
