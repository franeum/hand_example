from pythonosc import udp_client
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

client = udp_client.SimpleUDPClient("127.0.0.1", 6448)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for kind, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
                myhand = kind.classification[0].label
                # index finger tip
                one_hand = []
                for finger in hand_landmarks.landmark:
                    # one_hand.extend([max(0, min(1, finger.x)),
                    # max(0, min(1, finger.y))])
                    one_hand.append(max(0, min(1, finger.y)))

                print(myhand, one_hand)

                """
                if myhand == "Left":
                    client.send_message("/wek/inputs", one_hand)
                else:
                    client.send_message("/right", one_hand)
                """
                client.send_message("/wek/inputs", one_hand)

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        # cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
