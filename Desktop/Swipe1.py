import cv2
import mediapipe as mp
import pyautogui
import time
import math
import sys

# ==============================
# CONFIG
# ==============================
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Pointer
SMOOTH_ALPHA = 0.6
JITTER_THRESH = 3.0

# Swipe (INDEX + MIDDLE MERGED)
SWIPE_TRIGGER_DIST = 0.15
SWIPE_COOLDOWN = 0.5
MIN_HAND_SCALE = 0.15
MERGED_FINGER_THRESH = 0.05

# Scissors (Open App)
SCISSOR_OPEN_THRESH = 0.1  # Distance ratio to considered "Open"
SCISSOR_CLOSE_THRESH = 0.05 # Distance ratio to considered "Closed"
SCISSOR_COOLDOWN = 1.0

# Fist Toggle
FIST_COOLDOWN = 1.0

# ==============================
# INIT
# ==============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)
try:
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
except: pass

pyautogui.FAILSAFE = False

# ==============================
# STATE
# ==============================
prev_cursor = None
system_active = True
last_fist_time = 0

swipe_start_x = None
last_swipe_time = 0
swipe_detected = False

scissor_stage = 0 # 0: Idle, 1: Open Wide
last_scissor_time = 0

# ==============================
# HELPERS
# ==============================
def fingers_up(lm):
    # Tip < Pip means Tip is "higher" on screen (MP y coords)
    thumb = lm[4].x > lm[3].x 
    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y
    
    return thumb, index, middle, ring, pinky

def get_dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def apply_heat_wave_effect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return heatmap

def open_app():
    if sys.platform.startswith("win"):
        pyautogui.press("win")
    elif sys.platform.startswith("darwin"):
        pyautogui.hotkey("command", "space")
    else:
        pyautogui.press("super")

# ==============================
# MAIN LOOP
# ==============================
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # Process
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    
    # Visualize
    vis_frame = apply_heat_wave_effect(frame)
    h, w, _ = frame.shape
    now = time.time()

    if res.multi_hand_landmarks:
        lm = res.multi_hand_landmarks[0].landmark
        
        # Check Fingers
        _, idx_up, mid_up, ring_up, pinky_up = fingers_up(lm)
        
        # ==============================
        # 0. FIST TOGGLE (Global Pause)
        # ==============================
        if not idx_up and not mid_up and not ring_up and not pinky_up:
            if now - last_fist_time > FIST_COOLDOWN:
                system_active = not system_active
                last_fist_time = now
                print(f"System Toggled: {'ACTIVE' if system_active else 'PAUSED'}")

        # Draw Status
        status_text = "ACTIVE" if system_active else "PAUSED (FIST TO RESUME)"
        color = (0, 255, 0) if system_active else (0, 0, 255)
        cv2.putText(vis_frame, status_text, (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        
        # ==============================
        # ACTIVE MODE LOGIC
        # ==============================
        if system_active:
            idx_tip = lm[8]
            mid_tip = lm[12]
            
            # Distance between Index and Middle
            dist_idx_mid = get_dist(idx_tip, mid_tip)
            
            fingers_merged = dist_idx_mid < MERGED_FINGER_THRESH

            # --------------------------
            # 1. POINTER (Index Only)
            # --------------------------
            # Strict: Index UP, Middle DOWN (to avoid confusion with Scissors/Swipe)
            if idx_up and not mid_up and not ring_up and not pinky_up:
                sw, sh = pyautogui.size()
                pad = 0.1
                target_x = (idx_tip.x - pad) / (1 - 2*pad) * sw
                target_y = (idx_tip.y - pad) / (1 - 2*pad) * sh
                target_x = max(0, min(sw, target_x))
                target_y = max(0, min(sh, target_y))

                if prev_cursor is None:
                    prev_cursor = (target_x, target_y)

                sx = prev_cursor[0] * SMOOTH_ALPHA + target_x * (1 - SMOOTH_ALPHA)
                sy = prev_cursor[1] * SMOOTH_ALPHA + target_y * (1 - SMOOTH_ALPHA)

                if math.hypot(sx - prev_cursor[0], sy - prev_cursor[1]) > JITTER_THRESH:
                    pyautogui.moveTo(sx, sy, _pause=False)
                    prev_cursor = (sx, sy)
                
                cv2.putText(vis_frame, "POINTER", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Reset scissor stage if using pointer
                scissor_stage = 0 
            else:
                prev_cursor = None


            # --------------------------
            # 2. SCISSORS (Action: Open App)
            # --------------------------
            # Logic: Index & Middel UP. Ring & Pinky DOWN.
            # State Machine: Wide Open -> Close Shut
            
            if idx_up and mid_up and not ring_up and not pinky_up:
                
                # Check Open (Peace Sign)
                if dist_idx_mid > SCISSOR_OPEN_THRESH:
                    scissor_stage = 1
                    cv2.putText(vis_frame, "SCISSOR: OPEN...", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
                
                # Check Closed (Cut complete)
                elif dist_idx_mid < SCISSOR_CLOSE_THRESH:
                    if scissor_stage == 1:
                        if now - last_scissor_time > SCISSOR_COOLDOWN:
                            print("Scissors CUT -> Open App")
                            open_app()
                            last_scissor_time = now
                            scissor_stage = 0
                            cv2.putText(vis_frame, "APP LAUNCH!", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # If gesture broken, reset stage? 
                # Optional: keep it for a tiny bit to allow partial occlusions, but safer to reset.
                pass 


            # --------------------------
            # 3. SWIPE (Index + Middle Merged & Moving)
            # --------------------------
            # Only trigger if we aren't "Cutting"
            # If fingers are merged, it could be the end of a cut OR a swipe.
            # Differentation: Swipe requires MOVEMENT. Cut is an event.
            
            if idx_up and mid_up and fingers_merged and not ring_up and scissor_stage == 0:
                if swipe_start_x is None:
                    swipe_start_x = idx_tip.x
                    swipe_detected = False
                
                dx = idx_tip.x - swipe_start_x
                
                if not swipe_detected and now - last_swipe_time > SWIPE_COOLDOWN:
                    if dx < -SWIPE_TRIGGER_DIST:
                        pyautogui.press("right")
                        last_swipe_time = now
                        swipe_detected = True
                        cv2.putText(vis_frame, "NEXT >>", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    elif dx > SWIPE_TRIGGER_DIST:
                        pyautogui.press("left")
                        last_swipe_time = now
                        swipe_detected = True
                        cv2.putText(vis_frame, "<< PREV", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                cv2.putText(vis_frame, "SWIPE MODE", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                swipe_start_x = None

    # FPS
    curr_t = time.time()
    fps = 1 / (curr_t - prev_time) if prev_time else 0
    prev_time = curr_t
    cv2.putText(vis_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Heat Wave Gesture Control", vis_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
