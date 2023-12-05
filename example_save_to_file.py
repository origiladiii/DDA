import os
import cv2
from PIL import Image
import imagehash
import numpy as np
import multiprocessing as mp
import time
import json
import sys
from PIL import UnidentifiedImageError


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


def calculate_mse(image):
    img_array = np.array(image.resize((256, 256))).astype('float')
    mse = np.mean((img_array - np.mean(img_array)) ** 2)
    return mse


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


def save_media_paths(directory_path, file_list_path):
    media_files = gather_all_media_files(directory_path)
    with open(file_list_path, 'w') as f:
        json.dump(media_files, f)


def load_media_paths(file_list_path):
    with open(file_list_path, 'r') as f:
        return json.load(f)


def update_processed_pairs(pair, processed_pairs_file):
    with open(processed_pairs_file, 'a') as f:
        f.write(json.dumps(pair) + '\n')


def check_if_pair_processed(pair, processed_pairs_file):
    if not os.path.exists(processed_pairs_file):
        return False
    with open(processed_pairs_file, 'r') as f:
        processed_pairs = [json.loads(line.strip()) for line in f]
    return pair in processed_pairs


def load_media_data(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


# Function to save media data to file
def save_media_data(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def compute_and_store_media_data(media_file, media_data_file, calculate_mse=False):
    media_data = load_media_data(media_data_file)

    if media_file not in media_data:
        # Process images
        if media_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            try:
                with Image.open(media_file) as img:
                    media_data[media_file] = {
                        'phash': str(compute_phash(img)),
                        'histogram': get_histogram(img).tolist(),
                        'mse': calculate_mse(img) if calculate_mse else None
                    }
            except UnidentifiedImageError:
                print(f"Unable to identify image file: {media_file}")
                return None
        # Process videos
        elif media_file.lower().endswith(('.mp4', '.avi', '.mkv', '.flv', '.mov')):
            frames = get_video_frames(media_file, fps=10)
            if frames:
                frames_data = []
                for frame in frames:
                    frame_data = {
                        'phash': str(compute_phash(frame)),
                        'histogram': get_histogram(frame).tolist(),
                        'mse': calculate_mse(frame) if calculate_mse else None
                    }
                    frames_data.append(frame_data)
                media_data[media_file] = frames_data
            else:
                print(f"No frames extracted from {media_file}")
                return None

        save_media_data(media_data_file, media_data)

    return media_data.get(media_file)


def is_matching_frame_stored(ref_data, check_data, hash_size=8, mse_threshold=1000, hist_threshold=0.7):
    hist_score = compare_histograms(np.array(ref_data['histogram']), np.array(check_data['histogram']))
    if hist_score > hist_threshold:
        if ref_data['phash'] == check_data['phash']:
            return True
    return False


def remove_duplicates(media_files, media_data_file, hash_size=8, mse_threshold=1000, hist_threshold=0.7, max_cores=None):
    def file_exists(file_path):
        return os.path.exists(file_path)

    start_time = time.time()

    # Load the stored media data
    media_data = load_media_data(media_data_file)

    # Generate pairs of files to compare
    pairs = [(media_files[i], media_files[j]) for i in range(len(media_files)) for j in range(i + 1, len(media_files))]

    # Set the number of processes
    num_cores = mp.cpu_count() if max_cores is None else min(max_cores, mp.cpu_count())
    pool = mp.Pool(num_cores)

    # Process pairs in parallel
    func_args = [(pair, media_data, hash_size, mse_threshold, hist_threshold) for pair in pairs]
    results = pool.starmap(process_pair, func_args)

    # Handle the results and remove duplicates
    for result in results:
        if result:
            ref_file, check_file = result
            if file_exists(ref_file) and file_exists(check_file):
                # Delete the shorter video or lower quality image
                if ref_file.lower().endswith(('.mp4', '.avi', '.mkv', '.flv', '.mov')):
                    if get_video_duration(ref_file) > get_video_duration(check_file):
                        os.remove(check_file)
                    else:
                        os.remove(ref_file)
                else:
                    if get_image_quality_score(ref_file) > get_image_quality_score(check_file):
                        os.remove(check_file)
                    else:
                        os.remove(ref_file)

            # Update the media data file after removal
            compute_and_store_media_data(ref_file, media_data_file)
            compute_and_store_media_data(check_file, media_data_file)

    pool.close()
    pool.join()

    end_time = time.time()
    print(f"\nExecution time: {end_time - start_time}")


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


def process_pair(pair, media_data, hash_size, mse_threshold, hist_threshold):
    ref_file, check_file = pair
    try:
        if ref_file.lower().endswith(('.mp4', '.avi', '.mkv', '.flv', '.mov')):
            if check_file.lower().endswith(('.mp4', '.avi', '.mkv', '.flv', '.mov')):
                if are_videos_duplicates(ref_file, check_file, hash_size, mse_threshold, hist_threshold):
                    return ref_file, check_file
        else:
            ref_image = Image.open(ref_file)
            check_image = Image.open(check_file)
            if is_matching_frame(ref_image, check_image, hash_size, mse_threshold, hist_threshold):
                return ref_file, check_file
    except Exception as e:
        print(f"Error processing pair {ref_file}, {check_file}: {e}")
    return None


def process_directory(main_directory_path, media_data_file, max_cores, calculate_mse=False):
    print("Starting processing...")

    # Gather all media files in the directory
    all_media_files = gather_all_media_files(main_directory_path)
    print(f"Found {len(all_media_files)} media files for processing.")

    # Pre-process each media file to compute and store its hash and histogram data
    for media_file in all_media_files:
        compute_and_store_media_data(media_file, media_data_file, calculate_mse)

    # Call the remove_duplicates function to process and remove duplicates
    remove_duplicates(all_media_files, media_data_file, max_cores=max_cores)

    print(f"Finished processing all media files under: {main_directory_path}\n")


if __name__ == "__main__":
    media_data_file = 'media_data.json'
    process_directory('/home/nehoray/Documents/dupi', media_data_file, max_cores=3, calculate_mse=False)