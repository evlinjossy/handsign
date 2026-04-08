import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark

            fingers = []

            # Thumb
            if landmarks[4].x < landmarks[3].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Other 4 fingers
            tips = [8, 12, 16, 20]
            for tip in tips:
                if landmarks[tip].y < landmarks[tip - 2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = sum(fingers)
            # Simple sign mapping
            if total_fingers == 0:
                sign = "A"
            elif total_fingers == 5:
                sign = "B"
            elif total_fingers == 2:
                sign = "V"
            else:
                sign = ""

            cv2.putText(frame, f'Sign: {sign}', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.putText(frame, f'Fingers: {total_fingers}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()