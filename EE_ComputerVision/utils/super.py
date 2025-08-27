import cv2
import numpy as np
from utils.detectfaces import analyze_image, combine_faces, get_faces_front, get_faces_side, markup_fd
from utils.templatematch import analyze_template, markup_tm
from utils.sift import sift_analyze, sift_compare, sift_markup



# helper function for super image generation
def face_detection_sup(image, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels):

    # get faces
    faces_front, faces_front_flip = get_faces_front(image, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f)  
    faces_side, faces_side_flip = get_faces_side(image, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s) 
    faces = combine_faces(faces_front, faces_front_flip, faces_side, faces_side_flip)

    # process image
    faces_emotion_results, faces_gender_results = analyze_image(image, faces, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels)

    return faces, faces_emotion_results, faces_gender_results

# helper function for super image generation
def markup_tm_sup(image, method_choice, top_left, bottom_right, box_color, text_color, text_size, text_thick, box_thick, outline_bool, outline_color):
    # marks the method_choice on the rectangle 
    # creates a bounding box on the image over the coordinates provided by analyze_template

    # draw bounding box for template match
    cv2.rectangle(image, top_left, bottom_right, box_color, box_thick) # draw result rectangle

    # text inside the boxes
    text_corner = [top_left[0], top_left[1]]
    if outline_bool:
        image = cv2.putText(image, f'{method_choice}', org=(text_corner[0], text_corner[1]+20), fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

    image = cv2.putText(image, f'{method_choice}', org=(text_corner[0], text_corner[1]+20), fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color, thickness=text_thick)

# helper function for super image generation
def optical_flow_sup(img1, img2, total_random_colors=1000, color=None):

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
 
    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    image = img1.copy()
    next_image = img2.copy()

    # Take first image and find corners in it
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(gray_image, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(image)

    # Read second image
    gray_next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY)

    # Calculate Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        gray_image, gray_next_image, p0, None, **lk_params
    )
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    if color is None:
        # If color not specified, use random colors
        color_random = np.random.randint(0, 255, (total_random_colors, 3))
        # Draw the tracks
        # Random color assigned to each point
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a = int(a)
            b = int(b)
            c = int(c)
            d = int(d)
            mask = cv2.line(mask, (a, b), (c, d), color_random[j].tolist(), 2)
            next_image = cv2.circle(next_image, (a, b), 5, color_random[j].tolist(), -1)
    else:
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            a = int(a)
            b = int(b)
            c = int(c)
            d = int(d)
            mask = cv2.line(mask, (a, b), (c, d), color, 2)
            next_image = cv2.circle(next_image, (a, b), 5, color, -1)
    
    # add mask
    img = cv2.add(next_image, mask)
    return img

def super_image(image_list, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels, template, tm_methods, tm_methods_color, tm_method_choice, tm_num_results, sift_image, sift_ratio_threshold, sift_line_color, sift_point_color, sift_rside, sift_flag, box_color, box_thick, text_color, text_size, text_thick, text_color_tf, tm_box_color, tm_text_color, outline_bool, outline_color, full_output_path, fd_bool, tm_bool, of_bool, sift_bool, of_total_random_colors=1000, of_color=None):

    of_img_list = []
    if of_bool:
        # create a list of images with optical flow performed on them
        of_img_list.insert(0, image_list[0]) # optical_flow doesn't include first image in image_list, so need to include it here
        for i in range(len(image_list) -1):
            of_img = optical_flow_sup(image_list[i], image_list[i+1], of_total_random_colors, of_color)
            of_img_list.append(of_img)

    # Process each image in image_list and then markup at the end of the loop
    # if OPFLOW_IMG_BOOL was enabled, markup on images that were previously processed using optical flow
    # if not, then markup on the images that were extracted directly from the INPUT and have not been edited yet
    for x in range(len(image_list)):
        image = image_list[x].copy()
        if of_bool:
            final_image = of_img_list[x].copy()
        else:
            final_image = image_list[x].copy()
            
        if fd_bool:
            faces, faces_emotion_results, faces_gender_results = face_detection_sup(image, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels)

            markup_fd(final_image, faces, faces_emotion_results, faces_gender_results, box_color, box_thick, text_color, text_size, text_thick, text_color_tf, outline_bool, outline_color)


        if tm_bool:
            if tm_method_choice in tm_methods:
                rect_coord_list = analyze_template(image, template, tm_method_choice, tm_num_results)

                for j in range(len(rect_coord_list)):
                    markup_tm(final_image, tm_method_choice, rect_coord_list[j][0], rect_coord_list[j][1], tm_box_color, tm_text_color, text_size, text_thick, box_thick, outline_bool, outline_color)
            else:
                for i in range(len(tm_methods)):
                    rect_coord_list = analyze_template(image, template, tm_methods[i], tm_num_results)

                    for j in range(len(rect_coord_list)):
                        markup_tm_sup(final_image, tm_methods[i], rect_coord_list[j][0], rect_coord_list[j][1], tm_methods_color[i], tm_methods_color[i], text_size, text_thick, box_thick, outline_bool, outline_color)

        if sift_bool:
            good_matches, kp1, kp2 = sift_analyze(image, sift_image, sift_ratio_threshold, sift_rside)
            final_image = sift_markup(final_image, sift_image, sift_rside, sift_line_color, sift_point_color, sift_flag, good_matches, kp1, kp2)
      
        path = f'{full_output_path}_{x+1}.jpg'
        cv2.imwrite(path, final_image)
    
def super_video(video_path, update_every_x_frame, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels, template, tm_methods, tm_methods_color, tm_method_choice, sift_image, sift_ratio_threshold, sift_line_color, sift_point_color, sift_rside, sift_flag, box_color, box_thick, text_color, text_size, text_thick, text_color_tf, tm_box_color, tm_text_color, outline_bool, outline_color, full_output_path, fd_bool, tm_bool, of_bool, sift_bool, of_total_random_colors=1000, of_color=None):
     
    # read the video 
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))     # Width of frame
    frame_height = int(cap.get(4))    # Height of frame
    fps = cap.get(cv2.CAP_PROP_FPS)   # Frames per second

    if sift_bool: # setup for SIFT
        sift_width = sift_image.shape[1] # h, w, channel
        sift_height = sift_image.shape[0]

       # calculate new width for VideoWriter
        new_frame_width = frame_width + sift_width
        new_frame_height = frame_height
        if sift_height > frame_height:
            new_frame_height = sift_height

        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # BFMatcher with default params
        bf = cv2.BFMatcher()

        # Grayscale of sift_image
        sift_image_gray = cv2.cvtColor(sift_image.copy(), cv2.COLOR_BGR2GRAY)
   
    # initalize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # MP4 Codec
    out = cv2.VideoWriter(f'{full_output_path}' + '.mp4', fourcc, fps, (new_frame_width, new_frame_height))

    if of_bool: # setup for Optical Flow
        # Parameters for ShiTomasi corner detection
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
        # Parameters for Lucas Kanade optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
    
        if of_color is None:
            # Create random colors
            color_random = np.random.randint(0, 255, (of_total_random_colors, 3))
        
        # Take first frame and find corners in it
        ret, of_oldframe = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # reset back to first frame before entering loop
        of_oldframe_gray = cv2.cvtColor(of_oldframe, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(of_oldframe_gray, mask=None, **feature_params)
    
        # Create a mask image for drawing purposes
        mask = np.zeros_like(of_oldframe)
    
 
    # how often face data is updated (position, emotion, gender)
    # if a face is detected on the frame, the bounding box will stay in the same position in the video until face data is updated again
    frame_count = 0
    while cap.isOpened():
        # read new frame
        ret, frame = cap.read()
        if not ret: # break if video fails
            break

        # starting at zero, then every x frames update data for Face Detection, Template Match, and SIFT 
        if frame_count % update_every_x_frame == 0:  
            if fd_bool:
                # update faces
                faces_front, faces_front_flip = get_faces_front(frame, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f)  
                faces_side, faces_side_flip = get_faces_side(frame, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s) 
                faces = combine_faces(faces_front, faces_front_flip, faces_side, faces_side_flip)

                # update face results
                faces_emotion_results, faces_gender_results = analyze_image(frame, faces, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels)
            
            if tm_bool:
                if tm_method_choice not in tm_methods and tm_bool: # multi method
                    # update template match coordinates 
                    rect_coord_list = []
                    for i in range(len(tm_methods)):
                        temp = analyze_template(frame, template, tm_methods[i], num_results=1)
                        rect_coord_list = rect_coord_list + temp
                else:
                    # update template match coordinates 
                    rect_coord_list = analyze_template(frame, template, tm_method_choice, num_results=1)

            if sift_bool:
                # grayscale of current frame
                sift_frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
                
                # find the keypoints and descriptors with SIFT
                kp1, des1 = sift.detectAndCompute(sift_frame_gray, None)
                kp2, des2 = sift.detectAndCompute(sift_image_gray, None)

                if sift_rside:
                    matches = bf.knnMatch(des1, des2, k=2)
                else:
                    matches = bf.knnMatch(des2, des1, k=2)

                # apply ratio test
                good = []
                for m, n in matches:
                    if m.distance < sift_ratio_threshold*n.distance:
                        good.append([m])

        # every frame update data for Optical Flow       
        if of_bool: 
            of_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # calculate Optical Flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                of_oldframe_gray, of_frame_gray, p0, None, **lk_params
            )
            # select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        # markup fd every frame
        if fd_bool:
            # draw face bounding boxes every frame
            markup_fd(frame, faces, faces_emotion_results, faces_gender_results, box_color, box_thick, text_color, text_size, text_thick, text_color_tf, outline_bool, outline_color)

        # markup tm every frame
        if tm_bool:
            if tm_method_choice not in tm_methods and tm_bool: # multi method
                # draw template match bounding boxes every frame
                for x in range(len(rect_coord_list)):
                    markup_tm_sup(frame, tm_methods[x], rect_coord_list[x][0], rect_coord_list[x][1], tm_methods_color[x], tm_methods_color[x], text_size, text_thick, box_thick, outline_bool, outline_color)
            else:
                # draw template match bounding boxes every frame
                markup_tm(frame, tm_method_choice, rect_coord_list[0][0], rect_coord_list[0][1], tm_box_color, tm_text_color, text_size, text_thick, box_thick, outline_bool, outline_color)

        # markup of every frame
        if of_bool:
            if of_color is None:
                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a = int(a)
                    b = int(b)
                    c = int(c)
                    d = int(d)
                    cv2.line(mask, (a, b), (c, d), color_random[i].tolist(), 2)
                    cv2.circle(frame, (a, b), 5, color_random[i].tolist(), -1)
                frame = cv2.add(frame, mask)
            else:
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a = int(a)
                    b = int(b)
                    c = int(c)
                    d = int(d)
                    cv2.line(mask, (a, b), (c, d), of_color, 2)
                    cv2.circle(frame, (a, b), 5, of_color, -1)
                frame = cv2.add(frame, mask)
        
            # update the previous frame and previous points
            of_oldframe_gray = of_frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            # no features detected in the previous frame
            if p0 is None or len(p0) == 0:
                # set the features to the current frame
                # reset mask so that it does not fill up the video with a mess of colors 
                p0 = cv2.goodFeaturesToTrack(of_frame_gray, mask=None, **feature_params)
                mask = np.zeros_like(of_oldframe)

        # markup sift every frame
        if sift_bool:
            frame = sift_markup(frame, sift_image, sift_rside, sift_line_color, sift_point_color, sift_flag, good, kp1, kp2)

        out.write(frame)
        frame_count += 1

    # Release everything if job is finished
    cap.release()
    out.release()

def super_stream(camera_index, endkey, save_bool, update_every_x_frame, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels, template, tm_methods, tm_methods_color, tm_method_choice, sift_image, sift_ratio_threshold, sift_line_color, sift_point_color, sift_rside, sift_flag, box_color, box_thick, text_color, text_size, text_thick, text_color_tf, tm_box_color, tm_text_color, outline_bool, outline_color, full_output_path, fd_bool, tm_bool, of_bool, sift_bool, of_total_random_colors=1000, of_color=None):     
    # read the video 
    cap = cv2.VideoCapture(camera_index)
    frame_width = int(cap.get(3))     # Width of frame
    frame_height = int(cap.get(4))    # Height of frame
    fps = cap.get(cv2.CAP_PROP_FPS)   # Frames per second

    if sift_bool: # setup for SIFT
        sift_width = sift_image.shape[1] # h, w, channel
        sift_height = sift_image.shape[0]

        # calculate new width for VideoWriter
        new_frame_width = frame_width + sift_width
        new_frame_height = frame_height
        if sift_height > frame_height:
            new_frame_height = sift_height

        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # BFMatcher with default params
        bf = cv2.BFMatcher()

        # Grayscale of sift_image
        sift_image_gray = cv2.cvtColor(sift_image.copy(), cv2.COLOR_BGR2GRAY)
   
    if save_bool:
        # initalize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # MP4 Codec
        out = cv2.VideoWriter(f'{full_output_path}' + '.mp4', fourcc, fps, (new_frame_width, new_frame_height))

    if of_bool: # setup for Optical Flow
        # Parameters for ShiTomasi corner detection
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
        # Parameters for Lucas Kanade optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
    
        if of_color is None:
            # Create random colors
            color_random = np.random.randint(0, 255, (of_total_random_colors, 3))
        
        # Take first frame and find corners in it
        ret, of_oldframe = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # reset back to first frame before entering loop
        of_oldframe_gray = cv2.cvtColor(of_oldframe, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(of_oldframe_gray, mask=None, **feature_params)
    
        # Create a mask image for drawing purposes
        mask = np.zeros_like(of_oldframe)
    
 
    # how often face data is updated (position, emotion, gender)
    # if a face is detected on the frame, the bounding box will stay in the same position in the video until face data is updated again
    frame_count = 0
    while cap.isOpened():
        # read new frame
        ret, frame = cap.read()
        if not ret: # break if video fails
            break

        # starting at zero, then every x frames update data for Face Detection, Template Match, and SIFT 
        if frame_count % update_every_x_frame == 0:  
            if fd_bool:
                # update faces
                faces_front, faces_front_flip = get_faces_front(frame, face_detection_front, scaleFactor_f, minNeighbors_f, minSize_f, maxSize_f)  
                faces_side, faces_side_flip = get_faces_side(frame, face_detection_side, scaleFactor_s, minNeighbors_s, minSize_s, maxSize_s) 
                faces = combine_faces(faces_front, faces_front_flip, faces_side, faces_side_flip)

                # update face results
                faces_emotion_results, faces_gender_results = analyze_image(frame, faces, emotion_classifier, gender_classifier, emotion_offsets, gender_offsets, emotion_labels, gender_labels)
            
            if tm_bool:
                if tm_method_choice not in tm_methods and tm_bool: # multi method
                    # update template match coordinates 
                    rect_coord_list = []
                    for i in range(len(tm_methods)):
                        temp = analyze_template(frame, template, tm_methods[i], num_results=1)
                        rect_coord_list = rect_coord_list + temp
                else:
                    # update template match coordinates 
                    rect_coord_list = analyze_template(frame, template, tm_method_choice, num_results=1)

            if sift_bool:
                # grayscale of current frame
                sift_frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
                
                # find the keypoints and descriptors with SIFT
                kp1, des1 = sift.detectAndCompute(sift_frame_gray, None)
                kp2, des2 = sift.detectAndCompute(sift_image_gray, None)

                if sift_rside:
                    matches = bf.knnMatch(des1, des2, k=2)
                else:
                    matches = bf.knnMatch(des2, des1, k=2)

                # apply ratio test
                good = []
                for m, n in matches:
                    if m.distance < sift_ratio_threshold*n.distance:
                        good.append([m])

        # every frame update data for Optical Flow       
        if of_bool: 
            of_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # calculate Optical Flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                of_oldframe_gray, of_frame_gray, p0, None, **lk_params
            )
            # select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        # markup fd every frame
        if fd_bool:
            # draw face bounding boxes every frame
            markup_fd(frame, faces, faces_emotion_results, faces_gender_results, box_color, box_thick, text_color, text_size, text_thick, text_color_tf, outline_bool, outline_color)

        # markup tm every frame
        if tm_bool:
            if tm_method_choice not in tm_methods and tm_bool: # multi method
                # draw template match bounding boxes every frame
                for x in range(len(rect_coord_list)):
                    markup_tm_sup(frame, tm_methods[x], rect_coord_list[x][0], rect_coord_list[x][1], tm_methods_color[x], tm_methods_color[x], text_size, text_thick, box_thick, outline_bool, outline_color)
            else:
                # draw template match bounding boxes every frame
                markup_tm(frame, tm_method_choice, rect_coord_list[0][0], rect_coord_list[0][1], tm_box_color, tm_text_color, text_size, text_thick, box_thick, outline_bool, outline_color)

        # markup of every frame
        if of_bool:
            if of_color is None:
                # draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a = int(a)
                    b = int(b)
                    c = int(c)
                    d = int(d)
                    cv2.line(mask, (a, b), (c, d), color_random[i].tolist(), 2)
                    cv2.circle(frame, (a, b), 5, color_random[i].tolist(), -1)
                frame = cv2.add(frame, mask)
            else:
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a = int(a)
                    b = int(b)
                    c = int(c)
                    d = int(d)
                    cv2.line(mask, (a, b), (c, d), of_color, 2)
                    cv2.circle(frame, (a, b), 5, of_color, -1)
                frame = cv2.add(frame, mask)
        
            # update the previous frame and previous points
            of_oldframe_gray = of_frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            # no features detected in the previous frame
            if p0 is None or len(p0) == 0:
                # set the features to the current frame
                # reset mask so that it does not fill up the video with a mess of colors 
                p0 = cv2.goodFeaturesToTrack(of_frame_gray, mask=None, **feature_params)
                mask = np.zeros_like(of_oldframe)

        # markup sift every frame
        if sift_bool:
            frame = sift_markup(frame, sift_image, sift_rside, sift_line_color, sift_point_color, sift_flag, good, kp1, kp2)
        

        if save_bool:
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
    if save_bool:
        out.release()