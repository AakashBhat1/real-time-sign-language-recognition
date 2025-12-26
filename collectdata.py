import os
import cv2

# Auto-create folders A-Z if they don't exist
def create_image_dirs(base_dir='Image'):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        path = os.path.join(base_dir, ch)
        os.makedirs(path, exist_ok=True)

# Get current count of images in each directory
def get_image_counts(base_dir='Image'):
    return {ch.lower(): len(os.listdir(os.path.join(base_dir, ch))) for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}

def main():
    base_dir = 'Image'
    create_image_dirs(base_dir)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'a' to 'z' to save frames for that label.")
    print("Press 'ESC' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count = get_image_counts(base_dir)
        # Define region of interest
        roi = frame[40:400, 0:300]
        cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

        # Show windows
        cv2.imshow("Sign Language Data Collection", frame)
        cv2.imshow("ROI", roi)

        # Capture key press
        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # ESC key to exit
            break

        elif ord('a') <= key <= ord('z'):
            char = chr(key)
            save_path = os.path.join(base_dir, char.upper(), f"{count[char]}.png")
            cv2.imwrite(save_path, roi)
            print(f"Saved: {save_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


