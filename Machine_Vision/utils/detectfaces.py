import os
import cv2
import numpy as np 

# helper function
def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets

    x1 = x - x_off
    x2 = x + width + x_off
    y1 = y - y_off
    y2 = y + height + y_off

    if x1 < 0:
        x1 = 0
    
    if x2 < 0:
        x2 = 0
    
    if y1 < 0:
        y1 = 0

    if y2 < 0:
        y2 = 0

    return (x1, x2, y1, y2)

# helper function
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def get_faces_front(image, face_detection_front, scaleFactor, minNeighbors, minSize, maxSize):
    # image - image that has already been read using cv2.imread
    # face_detection_front - loaded model using cv2.CascadeClassifier(detection_model_front_path)
    # scaleFactor – how much the image size is reduced at each image scale
    # minNeighbors - how many neighbors each candidate rectangle should have to retain it. Higher value results in fewer detections
    # minSize - minimum possible object size
    # maxSize - maximum possible object size

    # returns 2 numpy arrays of face coordinates. The first array is the faces detected using frontal face detection, and the second array is the faces detected using frontal face detection but with the image being flipped horizontally beforehand.  

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')
  
    # detect faces
    faces_1 = face_detection_front.detectMultiScale(gray_image, scaleFactor, minNeighbors, 0, minSize, maxSize)
    
    # flip across y axis and detect
    gray_image_flip = cv2.flip(gray_image, 1)
    faces_2 = (face_detection_front.detectMultiScale(gray_image_flip, scaleFactor, minNeighbors, 0, minSize, maxSize))

    if len(faces_1) == 0:  # If faces_1 is empty
        faces_1 = None

    if len(faces_2) == 0:  # If faces_2 is empty
        faces_2 = None

    # for the faces detected in the horizontally flipped image, flip their coordinates so that they display bounding boxes correctly on the original image.
    # the x-coordinate will change, while the y-coordinate and height remain the same. 
    # new_x = img_width - x - width
    if faces_2 is not None:
        dimensions = image.shape # h, w, channel
        image_width = dimensions[1]
        for i in range(len(faces_2)):
            # new_x = img_width - x - width
            new_x = image_width - faces_2[i][0] - faces_2[i][2]
            faces_2[i][0] = new_x

    return faces_1, faces_2 

def get_faces_side(image, face_detection_side, scaleFactor, minNeighbors, minSize, maxSize):
    # image - image that has already been read using cv2.imread
    # face_detection_side - loaded model using cv2.CascadeClassifier(detection_model_side_path)
    # scaleFactor – how much the image size is reduced at each image scale
    # minNeighbors - how many neighbors each candidate rectangle should have to retain it. Higher value results in fewer detections
    # minSize - minimum possible object size
    # maxSize - maximum possible object size

    # returns 2 numpy arrays of face coordinates. The first array is the faces detected using side profile face detection, and the second array is the faces detected using side profile face detection but with the image being flipped horizontally beforehand.  

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    # detect faces
    faces_1 = face_detection_side.detectMultiScale(gray_image, scaleFactor, minNeighbors, 0, minSize, maxSize)

    # flip across y axis and detect
    gray_image_flip = cv2.flip(gray_image, 1)
    faces_2 = (face_detection_side.detectMultiScale(gray_image_flip, scaleFactor, minNeighbors, 0, minSize, maxSize))

    if len(faces_1) == 0:  # If faces is empty
        faces_1 = None

    if len(faces_2) == 0:  # If faces is empty
        faces_2 = None

    # for the faces detected in the flipped image, flip their coordinates so that they display bounding boxes correctly on the original image.
    # the x-coordinate will change, while the y-coordinate and height remain the same. 
    # new_x = img_width - x - width
    if faces_2 is not None:
        dimensions = image.shape # h, w, channel
        image_width = dimensions[1]
        for i in range(len(faces_2)):
            # new_x = img_width - x - width
            new_x = image_width - faces_2[i][0] - faces_2[i][2]
            faces_2[i][0] = new_x

    return faces_1, faces_2

def combine_faces(faces_front, faces_front_flip, faces_side, faces_side_flip):
    # faces_front - numpy array result of get_faces_front. Contains the faces detected using front face detection
    # faces_front_flip - numpy array result of get_faces_front. Contains the faces detected using front face detection using flipped image
    # faces_side - numpy array result of get_faces_side. Contains the faces detected using side profile face detection
    # faces_side_flip - numpy array result of get_faces_side. Contains the faces detected using side profile face detection using flipped image

    # returns a list of face coordinates for bounding boxes that do not overlap 

    # place in dictionary
    # faces_dict =
    # {
    #  'front':[[ 1 2 3 4], [1 2 3 4]]
    #  'front_flip':[[ 1 2 3 4], [1 2 3 4]]
    #  'side':[[ 1 2 3 4], [1 2 3 4]]
    #  'side_flip':[[ 1 2 3 4], [1 2 3 4]]
    # }
    faces_dict = {}
    faces_dict['front'] = faces_front
    faces_dict['front_flip'] = faces_front_flip
    faces_dict['side'] = faces_side
    faces_dict['side_flip'] = faces_side_flip

    # create new empty numpy array, and place all numpy arrays of face coordinates into it 
    faces = np.empty((0, 4), dtype=int)
    if faces_front is not None:
        faces = np.concatenate((faces, faces_front), axis=0)
    if faces_front_flip is not None:
        faces = np.concatenate((faces, faces_front_flip), axis=0)
    if faces_side is not None:
        faces = np.concatenate((faces, faces_side), axis=0)
    if faces_side_flip is not None:
        faces = np.concatenate((faces, faces_side_flip), axis=0)

    delete_list = [] # list of indexes that must be deleted from faces
    # itterate over numpy array and compare each element (set of coordinates for a face) to all other elements
    # remove face bounding boxes that are inside of another face bounding box 
    for i in range(len(faces)):
        # XYWH
        # X: represents the x-coordinate of the top-left corner of the bounding box
        # Y: represents the y-coordinate of the top-left corner of the bounding box
        # W: represents the width of the bounding box
        # H: represents the height of the bounding box
        x1 = faces[i][0]
        y1 = faces[i][1]
        w1 = faces[i][2]
        h1 = faces[i][3]
        # convert to xyxy format (xmin, ymin, xmax, ymax)
        box1_xmin, box1_ymin, box1_xmax, box1_ymax = x1, y1, x1 + w1, y1 + h1
        area_i = w1 * h1

        for j in range(i + 1, len(faces)):
            x2 = faces[j][0]
            y2 = faces[j][1]
            w2 = faces[j][2]
            h2 = faces[j][3]
            # convert to xyxy format (xmin, ymin, xmax, ymax)
            box2_xmin, box2_ymin, box2_xmax, box2_ymax =  x2, y2, x2 + w2, y2 + h2 
            area_j = w2 * h2 

            # check for overlap conditions
            # if box1 is not on the right, left, above, or below box 2, delete smaller box  
            if not (
            box1_xmax < box2_xmin or  # box1 is to the left of box2
            box1_xmin > box2_xmax or  # box1 is to the right of box2
            box1_ymax < box2_ymin or  # box1 is above box2
            box1_ymin > box2_ymax): # box1 is below box2

                if (area_i < area_j):
                    delete_list.append(i)
                else:
                    delete_list.append(j)

    delete_list = list(set(delete_list)) # remove duplicate indexes
    faces_clean = np.delete(faces, delete_list, axis=0)

    return faces_clean 

def analyze_image(image, faces, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels):
    # image - image that has already been read using cv2.imread
    # faces - numpy array of face coordinates
    # emotion_classifier - loaded emotion classification model using tensorflow.keras.models load_model(emotion_model_path, compile=False)
    # gender_classifier - loaded gender classification model using tensorflow.keras.models load_model(gender_model_path, compile=False)
    # emotion_offsets - important for emotion classification. Area for emotion classification extends beyond the range of the face bounding box by the offset
    # gender_offsets - important for gender classification. Area for gender classification extends beyond the range of the face bounding box by the offset 
    # emotion_labels - from datasets.py get_labels('fer2013')
    # gender_labels - from datasets.py get_labels('imdb')

    # returns two lists. faces_emotion_result, which is the determined emotion of each face. faces_gender_result, which is the determined gender of each face

    faces_emotion_results = []
    faces_gender_results = []

    if faces is None: # if no faces were detected, return empty lists
        return faces_emotion_results, faces_gender_results

    # input_shape[1:3] cuts out the first and last values in emotion/gender_classifier.input_shape
    # (None, 64, 64, 1) --> (64, 64)
    emotion_target_size = emotion_classifier.input_shape[1:3] 
    gender_target_size = gender_classifier.input_shape[1:3]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.squeeze(gray_image)
    gray_image = gray_image.astype('uint8')

    # draw bounding boxes on the image
    for (x, y, w, h) in faces:
        # convert face coordinates in xyxy format and apply an offset
        # the area for emotion and gender classification extends beyond the range of the bounding box by an offset
        x1, x2, y1, y2 = apply_offsets((x, y, w, h), emotion_offsets)
        gray_face_emo = gray_image[y1:y2, x1:x2]

        x1, x2, y1, y2 = apply_offsets((x, y, w, h), gender_offsets)
        gray_face_gen = gray_image[y1:y2, x1:x2]

        # shrink detected face so that classifiers can work properly 
        try:
            gray_face_emo = cv2.resize(gray_face_emo, (emotion_target_size))
            gray_face_gen = cv2.resize(gray_face_gen, (gender_target_size))

        except Exception as e:
            (e, 'Something went wrong with resize. Image may be too small. Text may not appear')
            continue

        # face preprocess
        gray_face_emo = preprocess_input(gray_face_emo, True)
        gray_face_emo = np.expand_dims(gray_face_emo, 0)
        gray_face_emo = np.expand_dims(gray_face_emo, -1)
        gray_face_gen = preprocess_input(gray_face_gen, True)
        gray_face_gen = np.expand_dims(gray_face_gen, 0)
        gray_face_gen = np.expand_dims(gray_face_gen, -1)

        # emotion prediction using grayscale image
        emotion_prediction = emotion_classifier.predict(gray_face_emo)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]

        # gender prediction using grayscale image
        gender_prediction = gender_classifier.predict(gray_face_gen)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]

        # text to be written on image 
        # text = f'{emotion_text}, {gender_text}'
        faces_emotion_results.append(emotion_text)
        faces_gender_results.append(gender_text)

    return faces_emotion_results, faces_gender_results

def markup_fd(image, faces, faces_emotion_results, faces_gender_results, box_color, box_thick, text_color, text_size, text_thick, text_color_tf, outline_bool, outline_color):

    for i in range(len(faces)):
        (x, y, w, h) = faces[i]

        # bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), box_color, box_thick)

        # text to be written on image 
        text = f'{faces_emotion_results[i]}, {faces_gender_results[i]}'

        # apply text for each face 
        if outline_bool:
            # text outline
            image = cv2.putText(image, text, org=(x-10,y-30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

        image = cv2.putText(image, text, org=(x-10,y-30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color, thickness=text_thick)

    # total faces detected text
    text_corner = [0, 30]
    text_corner[1] = int(text_corner[1] * text_size)
    if outline_bool:
            # text outline
            image = cv2.putText(image, f'Faces detected: {len(faces)}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

    image = cv2.putText(image, f'Faces detected: {len(faces)}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color_tf, thickness=text_thick)

def face_detection(img, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels, box_color, box_thick, text_color, text_size, text_thick, text_color_tf, outline_bool, outline_color, full_output_path):
    # Uses the get_faces_front, get_faces_side, and analyze_image functions together to create a list of images and their paths to be saved
    
    image = img.copy()

    # get faces
    faces_front, faces_front_flip = get_faces_front(image, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f)  
    faces_side, faces_side_flip = get_faces_side(image, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s) 
    faces = combine_faces(faces_front, faces_front_flip, faces_side, faces_side_flip)

    # process image
    faces_emotion_results, faces_gender_results = analyze_image(image, faces, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels)

    # mark image
    markup_fd(image, faces, faces_emotion_results, faces_gender_results, box_color, box_thick, text_color, text_size, text_thick, text_color_tf, outline_bool, outline_color)

    # write output image to .jpg and place in a folder
    path = f'{full_output_path}.jpg'

    return path, image


def face_detection_video(video_path, update_every_x_frame, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels, box_color, box_thick, text_color, text_size, text_thick, text_color_tf, outline_bool, outline_color, full_output_path):
    # uses the get_faces_front, get_faces_side, and analyze_image functions together to process a video and write a new video
     
    # read the video 
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))     # Width of frame
    frame_height = int(cap.get(4))    # Height of frame
    fps = cap.get(cv2.CAP_PROP_FPS)   # Frames per second

    # initalize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # MP4 Codec
    out = cv2.VideoWriter(f'{full_output_path}' + '.mp4', fourcc, fps, (frame_width, frame_height))
   
    # how often face data is updated (position, emotion, gender)
    # if a face is detected on the frame, the bounding box will stay in the same position in the video until face data is updated again
    frame_count = 0
    while cap.isOpened():
        # read new frame
        ret, frame = cap.read()
        if not ret: # break if video fails
            break

        if frame_count % update_every_x_frame == 0:  # starting at zero, then every x frames update faces data
            # get faces
            faces_front, faces_front_flip = get_faces_front(frame, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f)  
            faces_side, faces_side_flip = get_faces_side(frame, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s) 
            faces = combine_faces(faces_front, faces_front_flip, faces_side, faces_side_flip)

            # process frame
            faces_emotion_results, faces_gender_results = analyze_image(frame, faces, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels)
            
        # draw boxes every frame
        for i in range(len(faces)):
            (x, y, w, h) = faces[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, box_thick)

            # text to be written on frame 
            text = f'{faces_emotion_results[i]}, {faces_gender_results[i]}'

            # apply text for each face 
            if outline_bool:
                # text outline
                frame = cv2.putText(frame, text, org=(x,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

            frame = cv2.putText(frame, text, org=(x,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color, thickness=text_thick)

        # total faces detected text
        text_corner = [0, 30]
        text_corner[1] = int(text_corner[1] * text_size)
        if outline_bool:
            # text outline
            frame = cv2.putText(frame, f'Faces detected: {len(faces)}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

        frame = cv2.putText(frame, f'Faces detected: {len(faces)}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color_tf, thickness=text_thick)

        out.write(frame)
        frame_count += 1

    # Release everything if job is finished
    cap.release()
    out.release()


# def face_detection_stream(camera_index, update_every_x_frame, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels, box_color, box_thick, text_color, text_size, text_thick, text_color_tf, outline_bool, outline_color, full_output_path):
#     # uses the get_faces_front, get_faces_side, and analyze_image functions together to process a video and write a new video
     
#     # read the video 
#     cap = cv2.VideoCapture(camera_index)
#     frame_width = int(cap.get(3))     # Width of frame
#     frame_height = int(cap.get(4))    # Height of frame
#     fps = cap.get(cv2.CAP_PROP_FPS)   # Frames per second

#     # initalize VideoWriter
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # MP4 Codec
#     out = cv2.VideoWriter(f'{full_output_path}' + '.mp4', fourcc, fps, (frame_width, frame_height))
   
#     # how often face data is updated (position, emotion, gender)
#     # if a face is detected on the frame, the bounding box will stay in the same position in the video until face data is updated again
#     frame_count = 0
#     while cap.isOpened():
#         # read new frame
#         ret, frame = cap.read()
#         if not ret: # break if video fails
#             break

#         if frame_count % update_every_x_frame == 0:  # starting at zero, then every x frames update faces data
#             # get faces
#             faces_front, faces_front_flip = get_faces_front(frame, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f)  
#             faces_side, faces_side_flip = get_faces_side(frame, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s) 
#             faces = combine_faces(faces_front, faces_front_flip, faces_side, faces_side_flip)

#             # process frame
#             faces_emotion_results, faces_gender_results = analyze_image(frame, faces, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels)
            
#         # draw boxes every frame
#         for i in range(len(faces)):
#             (x, y, w, h) = faces[i]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, box_thick)

#             # text to be written on frame 
#             text = f'{faces_emotion_results[i]}, {faces_gender_results[i]}'

#             # apply text for each face 
#             if outline_bool:
#                 # text outline
#                 frame = cv2.putText(frame, text, org=(x,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

#             frame = cv2.putText(frame, text, org=(x,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color, thickness=text_thick)

#         # total faces detected text
#         text_corner = [0, 30]
#         text_corner[1] = int(text_corner[1] * text_size)
#         if outline_bool:
#             # text outline
#             frame = cv2.putText(frame, f'Faces detected: {len(faces)}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

#         frame = cv2.putText(frame, f'Faces detected: {len(faces)}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color_tf, thickness=text_thick)

#         out.write(frame)
#         frame_count += 1

#         cv2.imshow('Live Stream', frame)
#         # Check if 'X' button is clicked or 'q' key is pressed
#         key = cv2.waitKey(1) & 0xFF  # waitKey(1) for continuous frame updates
#         if key == ord('q') or cv2.getWindowProperty('Live Stream', cv2.WND_PROP_VISIBLE) < 1:
#             break

#     # Release everything if job is finished
#     cv2.destroyAllWindows()
#     cap.release()
#     out.release()

def face_detection_stream(camera_index, endkey, save_bool, update_every_x_frame, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels, box_color, box_thick, text_color, text_size, text_thick, text_color_tf, outline_bool, outline_color, full_output_path):
     
    # read the video 
    cap = cv2.VideoCapture(camera_index)
    frame_width = int(cap.get(3))     # Width of frame
    frame_height = int(cap.get(4))    # Height of frame
    fps = cap.get(cv2.CAP_PROP_FPS)   # Frames per second

    frame_count = 0


    if save_bool:
        # initalize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # MP4 Codec
        out = cv2.VideoWriter(f'{full_output_path}' + '.mp4', fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            # read new frame
            ret, frame = cap.read()
            if not ret: # break if video fails
                break

            if frame_count % update_every_x_frame == 0:  # starting at zero, then every x frames update faces data
                # get faces
                faces_front, faces_front_flip = get_faces_front(frame, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f)  
                faces_side, faces_side_flip = get_faces_side(frame, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s) 
                faces = combine_faces(faces_front, faces_front_flip, faces_side, faces_side_flip)

                # process frame
                faces_emotion_results, faces_gender_results = analyze_image(frame, faces, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels)
            
            # draw boxes every frame
            for i in range(len(faces)):
                (x, y, w, h) = faces[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, box_thick)

                # text to be written on frame 
                text = f'{faces_emotion_results[i]}, {faces_gender_results[i]}'

                # apply text for each face 
                if outline_bool:
                    # text outline
                    frame = cv2.putText(frame, text, org=(x,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

                frame = cv2.putText(frame, text, org=(x,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color, thickness=text_thick)

            # total faces detected text
            text_corner = [0, 30]
            text_corner[1] = int(text_corner[1] * text_size)
            if outline_bool:
                # text outline
                frame = cv2.putText(frame, f'Faces detected: {len(faces)}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

            frame = cv2.putText(frame, f'Faces detected: {len(faces)}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color_tf, thickness=text_thick)

            out.write(frame)
            frame_count += 1
                
            cv2.namedWindow('Live Stream', cv2.WINDOW_NORMAL)
            cv2.imshow('Live Stream', frame)
            # Check if 'X' button is clicked or endkey is pressed
            key = cv2.waitKey(1) & 0xFF  # waitKey(1) for continuous frame updates
            if key == ord(endkey) or cv2.getWindowProperty('Live Stream', cv2.WND_PROP_VISIBLE) < 1:
                break

        # Release everything if job is finished
        cv2.destroyAllWindows()
        cap.release()
        out.release()

    else:
        while cap.isOpened():
            # read new frame
            ret, frame = cap.read()
            if not ret: # break if video fails
                break

            if frame_count % update_every_x_frame == 0:  # starting at zero, then every x frames update faces data
                # get faces
                faces_front, faces_front_flip = get_faces_front(frame, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f)  
                faces_side, faces_side_flip = get_faces_side(frame, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s) 
                faces = combine_faces(faces_front, faces_front_flip, faces_side, faces_side_flip)

                # process frame
                faces_emotion_results, faces_gender_results = analyze_image(frame, faces, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels)
            
            # draw boxes every frame
            for i in range(len(faces)):
                (x, y, w, h) = faces[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, box_thick)

                # text to be written on frame 
                text = f'{faces_emotion_results[i]}, {faces_gender_results[i]}'

                # apply text for each face 
                if outline_bool:
                    # text outline
                    frame = cv2.putText(frame, text, org=(x,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

                frame = cv2.putText(frame, text, org=(x,y+20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color, thickness=text_thick)

            # total faces detected text
            text_corner = [0, 30]
            text_corner[1] = int(text_corner[1] * text_size)
            if outline_bool:
                # text outline
                frame = cv2.putText(frame, f'Faces detected: {len(faces)}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

            frame = cv2.putText(frame, f'Faces detected: {len(faces)}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color_tf, thickness=text_thick)

            frame_count += 1
                
            cv2.namedWindow('Live Stream', cv2.WINDOW_NORMAL)
            cv2.imshow('Live Stream', frame)
            # Check if 'X' button is clicked or endkey is pressed
            key = cv2.waitKey(1) & 0xFF  # waitKey(1) for continuous frame updates
            if key == ord(endkey) or cv2.getWindowProperty('Live Stream', cv2.WND_PROP_VISIBLE) < 1:
                break

        # Release everything if job is finished
        cv2.destroyAllWindows()
        cap.release()


  
