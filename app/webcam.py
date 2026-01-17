import cv2
from ultralytics import YOLO
import time

def main():
    # Load model
    print("Loading model...")
    model = YOLO('model/best.pt')
    print("✅ Model loaded successfully!")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("✅ Webcam opened successfully!")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error: Failed to grab frame")
            break
        
        # Run detection
        results = model(frame, conf=0.25, verbose=False)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Calculate FPS
        fps_counter += 1
        if fps_counter >= 10:
            fps_end_time = time.time()
            fps = 10 / (fps_end_time - fps_start_time)
            fps_start_time = time.time()
            fps_counter = 0
        
        # Display FPS and info
        cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Count detections
        num_detections = len(results[0].boxes)
        cv2.putText(annotated_frame, f'Detections: {num_detections}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Instructions
        cv2.putText(annotated_frame, 'Press Q to quit, S to save', (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Eco Guardian - Real-time Waste Detection', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            screenshot_count += 1
            filename = f'detection_screenshot_{screenshot_count}.jpg'
            cv2.imwrite(filename, annotated_frame)
            print(f"✅ Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Application closed successfully!")

if __name__ == "__main__":
    main()