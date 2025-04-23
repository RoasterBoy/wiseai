import cv2
import os

def extract_frames(video_path, output_folder):
    """
    Extracts a frame from the video every second and saves it to the output folder.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where frames will be saved.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps)  # Number of frames to skip to extract one frame per second

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break  # Break when the video ends

        # Save one frame every second
        if frame_count % interval == 0:
            frame_name = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_name, frame)
            extracted_count += 1
            print(f"Saved: {frame_name}")

        frame_count += 1

    cap.release()
    print(f"Extraction completed. {extracted_count} frames saved to {output_folder}.")

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ").strip()
    output_folder = input("Enter the output folder to save frames: ").strip()
    extract_frames(video_path, output_folder)
