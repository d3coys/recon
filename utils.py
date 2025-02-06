import os
import tensorflow as tf
import cv2
import numpy as np

def load_model(model_path="models/truthfulness_model.h5"):
    """Load trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train it first!")
    return tf.keras.models.load_model(model_path)

def allowed_file(filename):
    """Check if file is a valid video format"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov', 'mkv'}

import cv2
import numpy as np

def draw_hud(frame, results, total_scores):
    """Draws the detection results, truthfulness score, progress bar, and conclusion on the frame."""
    
    avg_truthfulness = np.mean(total_scores) if total_scores else 0.0

    for face, truthfulness_score in results:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        # Determine progress bar color based on truthfulness score
        if truthfulness_score >= 0.71:
            bar_color = (0, 255, 0)  # Green for high truthfulness
        elif truthfulness_score >= 0.41:
            bar_color = (0, 255, 255)  # Yellow for uncertain
        else:
            bar_color = (0, 0, 255)  # Red for deception

        # Draw bounding box around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Define background box dimensions
        overlay_x, overlay_y = x, y + h + 30
        overlay_w, overlay_h = w, 150  # Adjusted height for proper text spacing

        # Draw black background for text and progress bar
        cv2.rectangle(frame, (overlay_x, overlay_y), 
                      (overlay_x + overlay_w, overlay_y + overlay_h), (0, 0, 0), -1)

        # Draw dynamic progress bar
        bar_x_start = overlay_x + 10
        bar_x_end = overlay_x + overlay_w - 10
        bar_y_start = overlay_y + 10
        bar_y_end = bar_y_start + 10
        bar_width = int((bar_x_end - bar_x_start) * truthfulness_score)

        cv2.rectangle(frame, (bar_x_start, bar_y_start),
                      (bar_x_start + bar_width, bar_y_end), bar_color, -1)

        # Proper spacing for text
        text_start_y = bar_y_end + 25

        # Display truthfulness score
        cv2.putText(frame, f"Truthfulness : {truthfulness_score:.2f}",
                    (overlay_x + 20, text_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display total accumulated score (average of all truthfulness scores)
        cv2.putText(frame, f"Total Score : {avg_truthfulness:.2f}",
                    (overlay_x + 20, text_start_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Determine conclusion text and color
        if avg_truthfulness >= 0.71:
            conclusion = "Truthful"
            conclusion_color = (0, 255, 0)  # Green
        elif avg_truthfulness >= 0.41:
            conclusion = "Uncertain"
            conclusion_color = (0, 255, 255)  # Yellow
        else:
            conclusion = "Deception"
            conclusion_color = (0, 0, 255)  # Red

        # Display conclusion
        cv2.putText(frame, f"Conclusion : {conclusion}",
                    (overlay_x + 20, text_start_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, conclusion_color, 2)

        # Display developer credit
        cv2.putText(frame, "By Opposite6890",
                    (overlay_x + 20, text_start_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame
