import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Camera initialization with fallback for old USB webcams
def init_camera():
    for backend in [cv2.CAP_VFW, cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
        for index in [0, 1, 2]:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                print(f"Camera opened with index {index}, backend {backend}")
                return cap
    raise RuntimeError("No working camera found")

cap = init_camera()

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.8)
draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

# Debounce timers
last_click_time = 0
last_ss_time = 0

def finger_extended(lm, tip, pip):
    return lm[tip].y < lm[pip].y

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            lm = hand.landmark
            draw.draw_landmarks(frame, hand)

            index_up = finger_extended(lm, 8, 6)
            middle_up = finger_extended(lm, 12, 10)
            ring_up = finger_extended(lm, 16, 14)
            pinky_up = finger_extended(lm, 20, 18)
            thumb_up = lm[4].x < lm[3].x

            # Cursor move
            if index_up and not (middle_up or ring_up or pinky_up):
                x = max(5, lm[8].x * screen_w)
                y = max(5, lm[8].y * screen_h)
                pyautogui.moveTo(x, y)

            # Left click (debounced)
            dist_index_thumb = math.hypot((lm[8].x - lm[4].x) * screen_w,
                                          (lm[8].y - lm[4].y) * screen_h)
            if dist_index_thumb < 40 and time.time() - last_click_time > 0.3:
                pyautogui.click()
                last_click_time = time.time()
                cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Right click (debounced)
            dist_middle_thumb = math.hypot((lm[12].x - lm[4].x) * screen_w,
                                           (lm[12].y - lm[4].y) * screen_h)
            if dist_middle_thumb < 40 and time.time() - last_click_time > 0.3:
                pyautogui.rightClick()
                last_click_time = time.time()
                cv2.putText(frame, "Right Click", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Scroll
            if index_up and middle_up and ring_up and pinky_up and thumb_up:
                palm_y = lm[0].y * screen_h
                if palm_y > screen_h / 2:
                    pyautogui.scroll(-50)
                    cv2.putText(frame, "Scroll Down", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    pyautogui.scroll(50)
                    cv2.putText(frame, "Scroll Up", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Screenshot gesture: palm open sideways + swipe left→right
            if index_up and middle_up and ring_up and pinky_up and thumb_up:
                # Check horizontal movement of wrist (landmark 0)
                wrist_x = lm[0].x * screen_w
                if wrist_x > screen_w * 0.7 and time.time() - last_ss_time > 2:
                    pyautogui.screenshot("gesture_screenshot.png")
                    last_ss_time = time.time()
                    cv2.putText(frame, "Screenshot Taken", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
