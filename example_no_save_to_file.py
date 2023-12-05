import os
import cv2
from PIL import Image
import imagehash
import numpy as np
import multiprocessing
import time
import sys
def compute_phash(image, hash_size=8):
    return imagehash.phash(image, hash_size)


def get_histogram(image):
    img_array = np.array(image)
    if len(img_array.shape) == 2:  # Check if the image is grayscale
        return cv2.calcHist([img_array], [0], None, [256], [0, 256])
    else:
        return cv2.calcHist([img_array], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])


def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def is_matching_frame(ref_frame, check_frame, hash_size=8, mse_threshold=1000, hist_threshold=0.7):
    # Check if both images have the same number of channels
    if len(ref_frame.split()) != len(check_frame.split()):
        return False

    hist1 = get_histogram(ref_frame)
    hist2 = get_histogram(check_frame)

    hist_score = compare_histograms(hist1, hist2)
    if hist_score > hist_threshold:
        if compute_phash(ref_frame, hash_size) == compute_phash(check_frame, hash_size):
            mse_val = np.sum(
                (np.array(ref_frame.resize((256, 256))).astype("float") - np.array(check_frame.resize((256, 256))).astype(
                    "float")) ** 2)
            mse_val /= float(ref_frame.size[0] * ref_frame.size[1])
            if mse_val < mse_threshold:
                return True
    return False


def get_video_frames(video_file, fps=16):
    frames = []
    try:
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_FPS, fps)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_frame = Image.fromarray(frame)
            if is_valid_image(img_frame):
                frames.append(img_frame)

        cap.release()
    except Exception as e:
        print(f"Error extracting frames from {video_file}: {e}")
        return None

    return frames


def are_videos_duplicates(ref_video, check_video, hash_size=8, mse_threshold=1000, hist_threshold=0.7):
    ref_duration = get_video_duration(ref_video)
    check_duration = get_video_duration(check_video)

    if ref_duration > 1.5 * check_duration or check_duration > 1.5 * ref_duration:
        return False

    shorter_video, longer_video = (ref_video, check_video) if ref_duration <= check_duration else (check_video, ref_video)

    shorter_video_frames = get_video_frames(shorter_video, fps=20)
    longer_video_frames = get_video_frames(longer_video, fps=20)

    if shorter_video_frames is None or longer_video_frames is None:
        print(f"Error processing one of the videos: {shorter_video} or {longer_video}. Skipping comparison.")
        return False

    match_index = -1
    for i, frame in enumerate(shorter_video_frames):
        if any(is_matching_frame(frame, check_frame, hash_size, mse_threshold, hist_threshold) for check_frame in longer_video_frames):
            match_index = i
            break

    if match_index == -1:
        return False

    for ref_frame, check_frame in zip(shorter_video_frames[match_index:], longer_video_frames[match_index:]):
        if not is_matching_frame(ref_frame, check_frame, hash_size, mse_threshold, hist_threshold):
            return False

    return True



def get_video_duration(video_file):
    try:
        cap = cv2.VideoCapture(video_file)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        return duration
    except Exception as e:
        print(f"Error getting duration of {video_file}: {e}")
        return 0


def get_image_quality_score(image_path):
    with Image.open(image_path) as img:
        return img.width * img.height


def print_progress(iteration, total, prefix='', length=150, fill='█', print_end='\r'):
    """
    Print a progress bar to console
    iteration : current iteration (Int)
    total     : total iterations (Int)
    prefix    : prefix string (Str)
    length    : bar length (Int)
    fill      : bar fill character (Str)
    print_end : end character (e.g. '\r', '\n') (Str)
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% Complete', end=print_end, flush=True)  # Add flush=True here


def remove_duplicates(media_files, hash_size=8, mse_threshold=1000, hist_threshold=0.7):
    def file_exists(file_path):
        return os.path.exists(file_path)

    start_time = time.time()

    # media_files = [os.path.join(root, file) for root, _, files in os.walk(directory_path) for file in files if
    #                file.lower().endswith(
    #                    ('.mp4', '.avi', '.mkv', '.flv', '.mov', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]

    deleted_files = set()  # to track deleted files

    total_files = len(media_files)
    for idx, ref_file in enumerate(media_files):
        # Print progress
        print_progress(idx + 1, total_files, prefix='Progress:', length=150)
        sys.stdout.flush()  # Flush the console buffer

        if ref_file in deleted_files:
            continue  # skip if the file was already deleted

        if not file_exists(ref_file):
            continue  # skip if file doesn't exist

        if ref_file.lower().endswith(('.mp4', '.avi', '.mkv', '.flv', '.mov')):
            for check_file in media_files[idx + 1:]:
                if check_file in deleted_files:
                    continue
                if not file_exists(check_file):
                    continue
                if check_file.lower().endswith(('.mp4', '.avi', '.mkv', '.flv', '.mov')):
                    if are_videos_duplicates(ref_file, check_file, hash_size, mse_threshold, hist_threshold):
                        # Delete the shorter video
                        if get_video_duration(ref_file) > get_video_duration(check_file):
                            os.remove(check_file)
                            deleted_files.add(check_file)  # track the deleted file
                        else:
                            os.remove(ref_file)
                            deleted_files.add(ref_file)
                            break
        else:
            ref_image_quality = get_image_quality_score(ref_file)
            for check_file in media_files[idx + 1:]:
                # Check the validity of the images before processing
                if not (is_valid_image(ref_file) and is_valid_image(check_file)):
                    continue  # skip this iteration if either image is invalid

                if check_file in deleted_files:
                    continue
                if not file_exists(check_file):
                    continue
                if not check_file.lower().endswith(('.mp4', '.avi', '.mkv', '.flv', '.mov')):
                    check_image = Image.open(check_file)
                    ref_image = Image.open(ref_file)
                    if is_matching_frame(ref_image, check_image, hash_size, mse_threshold, hist_threshold):
                        # Delete the image with worse quality
                        if ref_image_quality > get_image_quality_score(check_file):
                            os.remove(check_file)
                            deleted_files.add(check_file)  # track the deleted file
                        else:
                            os.remove(ref_file)
                            deleted_files.add(ref_file)
                            break

    end_time = time.time()
    print(f"\nExecution time : {end_time - start_time}")


def is_valid_image(input):
    try:
        if isinstance(input, str):  # Input is a file path
            with Image.open(input) as img:
                img.verify()  # verify that it is an image
        elif isinstance(input, Image.Image):  # Input is an Image object
            input.verify()
        else:
            return False
        return True
    except Exception:
        return False


def gather_all_media_files(directory_path):
    all_media_files = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mkv', '.flv', '.mov', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                media_path = os.path.join(root, file)
                all_media_files.append(media_path)
                print(f"Found media: {media_path}")

    return all_media_files


def process_directory(main_directory_path):
    print("go to gather_all_media_files")
    all_media_files = gather_all_media_files(main_directory_path)
    print(f"Processing all media files under: {main_directory_path}")
    remove_duplicates(all_media_files)
    print(f"Finished processing all media files under: {main_directory_path}\n")


if __name__ == "__main__":
    process_directory('/home/nehoray/Documents/dupi')
