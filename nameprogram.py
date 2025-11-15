import cv2
import mediapipe as mp
import numpy as np
import time
import ast
import json
from dataanalysis import AttentionAnalyzer, display_suggestions, save_session_report


file_name='totaltime.txt'
attention_file = 'attention_data.json'


def save_value(input_value,filename):
    with open(filename,'w') as f:
        f.write(str(input_value))

def load_value(filename):
    with open(filename,'r') as f:
        read=f.read()
    return read

def save_attention_data(data, filename):
    if len(data) > 10:
        data = data[-10:]
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving attention data: {e}")
        return False

def load_attention_data(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return []
    
def generate_detailed_report():
    analyzer = AttentionAnalyzer()
    analyzer.load_historical_data('attention_data.json')
    current_session_data = {
        "session_time": total_elapsed_time,
        "focus_time": total_focus_time,
        "max_attention_span": max_attention_span,
        "distraction_count": distraction_count,
        "focus_percentage": (total_focus_time / total_elapsed_time * 100) if total_elapsed_time > 0 else 0
    }
    report = analyzer.generate_session_report(current_session_data)
    report_filename = save_session_report(report)
    suggestions = analyzer.generate_suggestions()
    display_suggestions(suggestions)
    recommended_duration = analyzer.calculate_optimal_session_duration()
    minutes = recommended_duration // 60
    print(f"\nRECOMMENDED NEXT SESSION DURATION: {minutes} minutes")
    
    return report_filename
    
attention_history = load_attention_data(attention_file)

try:
    values=ast.literal_eval(load_value(file_name))
    print('Loaded values:', values)
except:
    print('Creating a new file...')
    values={}
    

mp_face_mesh = mp.solutions.face_mesh
face_mash = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
start_time = time.time()
running = True
total_elapsed_time = 0.0
acc_elapsed_time=float(load_value(file_name))

focus_history = []  
max_attention_span = 0  
current_focus_streak = 0  
last_focus_check = time.time()
focus_threshold = 5.0  
distraction_count = 0  
total_focus_time = 0  

previous_angles = None
movement_threshold = 5.0
movement_buffer = []  
movement_buffer_size = 3
instant_movement_limit = 8.0    
extreme_angle_limit = 20.0

def is_focused(current_angles, previous_angles, movement_buffer):
    if previous_angles is None:
        return True  
    
    if current_angles is not None and previous_angles is not None:
        movement_x = abs(current_angles[0] - previous_angles[0])
        movement_y = abs(current_angles[1] - previous_angles[1])
        movement_z = abs(current_angles[2] - previous_angles[2])
        
        if movement_x > instant_movement_limit or movement_y > instant_movement_limit + 3 or movement_z > instant_movement_limit:     
            return False    
        if abs(current_angles[0]) > extreme_angle_limit or abs(current_angles[1]) > extreme_angle_limit + 5 or abs(current_angles[2]) > extreme_angle_limit:
            return False
        total_movement = movement_x + movement_y + movement_z
        movement_buffer.append(total_movement)
        if len(movement_buffer) > movement_buffer_size:
            movement_buffer.pop(0)
        
        avg_movement = sum(movement_buffer) / len(movement_buffer) if movement_buffer else 0
        is_focused_result = avg_movement < movement_threshold       
        return is_focused_result
    
    return True

def update_focus_streak(is_focused_state, current_time):
    global current_focus_streak, max_attention_span, last_focus_check, distraction_count, total_focus_time
    
    time_since_last_check = current_time - last_focus_check
    last_focus_check = current_time
    
    if is_focused_state:
        current_focus_streak += time_since_last_check
        total_focus_time += time_since_last_check
        if current_focus_streak > max_attention_span:
            max_attention_span = current_focus_streak
    else:
        if current_focus_streak > focus_threshold:
            distraction_count += 1
        current_focus_streak = 0


while cap.isOpened():
    success, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mash.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    if running:
        total_elapsed_time = time.time() - start_time
    hours = int(total_elapsed_time // 3600)
    minutes = int((total_elapsed_time % 3600) // 60)
    seconds = int(total_elapsed_time % 60)
    stopwatch_text = f"Timer: {hours:02d}:{minutes:02d}:{seconds:02d}"
    cv2.putText(image, stopwatch_text, (375, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    current_angles = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])
            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, qx, qy, qz = cv2.RQDecomp3x3(rmat)
            x_angle = angles[0] * 360
            y_angle = angles[1] * 360
            z_angle = angles[2] * 360
            current_angles = [x_angle, y_angle, z_angle]

            if y_angle < -10:
                text = "Looking Left"  
            elif y_angle > 10:
                text = "Looking Right"
            elif x_angle < -10:
                text = "Looking Down"   
            elif x_angle > 10:
                text = "Looking Up"
            else:
                text = "Forward"
            is_focused_state = is_focused(current_angles, previous_angles, movement_buffer)
            update_focus_streak(is_focused_state, time.time())

            focus_color = (0, 255, 0) if is_focused_state else (0, 0, 255)
            focus_status = "FOCUSED" if is_focused_state else "DISTRACTED"

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y_angle * 10), int(nose_2d[1] - x_angle * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)
            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x_angle, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "y: " + str(np.round(y_angle, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "z: " + str(np.round(z_angle, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, focus_status, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, focus_color, 2)
            end= time.time()
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                      landmark_drawing_spec=drawing_spec,)
    
    previous_angles = current_angles if current_angles is not None else previous_angles

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        total_elapsed_time=total_elapsed_time
        acc_elapsed_time=float(load_value(file_name))
        acc_elapsed_time+=total_elapsed_time
        save_value(str(acc_elapsed_time),file_name)
        session_data = {
            "total_time": total_elapsed_time,
            "focus_time": total_focus_time,
            "max_attention_span": max_attention_span,
            "distraction_count": distraction_count,
            "focus_percentage": (total_focus_time / total_elapsed_time * 100) if total_elapsed_time > 0 else 0
        }
        attention_history.append(session_data)
        save_attention_data(attention_history, attention_file)
        report_file = generate_detailed_report()
        print(f"Detailed report saved to: {report_file}")
        break
    if cv2.waitKey(5) & 0xFF == ord('r'):
        session_data = {
            "total_time": total_elapsed_time,
            "focus_time": total_focus_time,
            "max_attention_span": max_attention_span,
            "distraction_count": distraction_count,
            "focus_percentage": (total_focus_time / total_elapsed_time * 100) if total_elapsed_time > 0 else 0
        }
        attention_history.append(session_data)
        save_attention_data(attention_history, attention_file)
        with open(file_name,'w') as f:
            f.write("0")
        break
    
cap.release()
cv2.destroyAllWindows()