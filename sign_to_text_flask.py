import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import time

# Load your new ASL Alphabet model
model = load_model("asl_model.h5")  # Make sure this is the downloaded model
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'space', 'del', 'nothing']

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# State variables
prev_letter = ""
sentence = ""
last_update = time.time()
update_interval = 1.5  # seconds

def get_frame():
    global prev_letter, sentence, last_update

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, sentence

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            
            # Get bounding box with 30% padding
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            width, height = max(x_coords)-min(x_coords), max(y_coords)-min(y_coords)
            
            x1 = max(int(min(x_coords) - width * 0.3), 0)
            y1 = max(int(min(y_coords) - height * 0.3), 0)
            x2 = min(int(max(x_coords) + width * 0.3), w)
            y2 = min(int(max(y_coords) + height * 0.3), h)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Preprocess like in training
            resized = cv2.resize(roi, (64, 64))
            normalized = resized.astype('float32') / 255.0
            reshaped = np.expand_dims(normalized, axis=0)

            # Predict
            prediction = model.predict(reshaped, verbose=0)[0]
            confidence = np.max(prediction)
            predicted_idx = np.argmax(prediction)
            predicted_class = labels[predicted_idx]

            # Only accept high-confidence predictions
            if confidence > 0.9:  # Increased threshold
                current_time = time.time()
                
                # Add to sentence only if stable prediction
                if predicted_class == prev_letter:
                    if (current_time - last_update) > update_interval:
                        if predicted_class == 'space':
                            sentence += ' '
                        elif predicted_class == 'del':
                            sentence = sentence[:-1]
                        elif predicted_class != 'nothing':
                            sentence += predicted_class
                        last_update = current_time
                else:
                    last_update = current_time
                
                prev_letter = predicted_class

                # Display prediction
                cv2.putText(frame, f'{predicted_class} ({confidence:.2f})',
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Uncertain',
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display sentence with background
    cv2.rectangle(frame, (10, 10), (600, 60), (0, 0, 0), -1)
    cv2.putText(frame, f'Text: {sentence}', (20, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes(), sentence

def add_space():
    global sentence
    sentence += " "

def reset_sentence():
    global sentence
    sentence = ""