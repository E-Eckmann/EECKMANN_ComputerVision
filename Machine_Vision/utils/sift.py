import cv2

# helper function
# returns good SIFT matches and two sets of keypoints, one for each image
def sift_analyze(other_image, sift_image, ratio_threshold, rside):
    # Compare SIFT of first image to the second image 
    other_image_gray = cv2.cvtColor(other_image.copy(), cv2.COLOR_BGR2GRAY)
    sift_image_gray = cv2.cvtColor(sift_image.copy(), cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(other_image_gray, None)
    kp2, des2 = sift.detectAndCompute(sift_image_gray, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    if rside:
        matches = bf.knnMatch(des1, des2, k=2)
    else:
        matches = bf.knnMatch(des2, des1, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold*n.distance:
            good_matches.append([m])
    
    return good_matches, kp1, kp2

# helper function
# returns the final markedup image for the SIFT feature
def sift_markup(other_image, sift_image, rside, line_color, point_color, sift_flag, good_matches, kp1, kp2):
    if rside:
        image = cv2.drawMatchesKnn(other_image, kp1, sift_image, kp2, good_matches, outImg=None, matchColor=line_color, singlePointColor=point_color, matchesMask=None, flags=sift_flag)
    else:
        image = cv2.drawMatchesKnn(sift_image, kp2, other_image, kp1, good_matches, outImg=None, matchColor=line_color, singlePointColor=point_color, matchesMask=None, flags=sift_flag)

    return image

# returns a path and an image
def sift_compare(other_image, sift_image, ratio_threshold, line_color, point_color, rside, full_output_path, sift_flag=0): 
# other_image - Training image. The image that sift_image is compared to.
# sift_image - Query image. The image that will be compared to other_image.
# ratio_threshold - Used to discard ambiguous SIFT matches. Higher values lead to more SIFT matches detected, lower values lead to less SIFT matches detected.
# line_color - Color for lines that connect SIFT matches.
# point_color - Color for points that mark the location of SIFT features.
# sift_flag - Flag for cv2.drawMatchesKnn()
# rside - If True, then sift_image will appear to the right of other_image in the resulting image. If False, sift_image will appear to the left.
# full_output_path - The output path for the image.

    good_matches, kp1, kp2 = sift_analyze(other_image, sift_image, ratio_threshold, rside)
    final_image = sift_markup(other_image, sift_image, rside, line_color, point_color, sift_flag, good_matches, kp1, kp2)
   
    path = f'{full_output_path}.jpg'
    return path, final_image

def sift_compare_video(video_path, update_every_x_frame, sift_image, ratio_threshold, line_color, point_color, rside, full_output_path, sift_flag=0):
    # get width of sift image
    sift_width = sift_image.shape[1] # h, w, channel
    sift_height = sift_image.shape[0]

    # read the video 
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))     # Width of frame
    frame_height = int(cap.get(4))    # Height of frame
    fps = cap.get(cv2.CAP_PROP_FPS)   # Frames per second

    # calculate new width for VideoWriter
    new_frame_width = frame_width + sift_width
    new_frame_height = frame_height
    if sift_height > frame_height:
        new_frame_height = sift_height

    # initalize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # MP4 Codec
    out = cv2.VideoWriter(f'{full_output_path}' + '.mp4', fourcc, fps, (new_frame_width, new_frame_height))

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    # Grayscale of sift_image
    sift_image_gray = cv2.cvtColor(sift_image.copy(), cv2.COLOR_BGR2GRAY)


    frame_count = 0
    while cap.isOpened():
        # read new frame
        ret, frame = cap.read()
        if not ret: # break if video fails
            break
        # update data every x frames
        if frame_count % update_every_x_frame == 0:
            # grayscale of current frame
            frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(frame_gray, None)
            kp2, des2 = sift.detectAndCompute(sift_image_gray, None)

            if rside:
                matches = bf.knnMatch(des1, des2, k=2)
            else:
                matches = bf.knnMatch(des2, des1, k=2)

            # Apply ratio test
            good = []
            for m, n in matches:
                if m.distance < ratio_threshold*n.distance:
                    good.append([m])
        
        outframe = sift_markup(frame, sift_image, rside, line_color, point_color, sift_flag, good, kp1, kp2)

        out.write(outframe)
        frame_count += 1

    # Release everything if job is finished
    cap.release()
    out.release()

def sift_compare_stream(camera_index, endkey, save_bool, update_every_x_frame, sift_image, ratio_threshold, line_color, point_color, rside, full_output_path, sift_flag=0):
    # get width of sift image
    sift_width = sift_image.shape[1] # h, w, channel
    sift_height = sift_image.shape[0]

    # read the video 
    cap = cv2.VideoCapture(camera_index)
    frame_width = int(cap.get(3))     # Width of frame
    frame_height = int(cap.get(4))    # Height of frame
    fps = cap.get(cv2.CAP_PROP_FPS)   # Frames per second

    # calculate new dimensions with border
    frame_width = frame_width + sift_width
    if sift_height > frame_height:
        frame_height = frame_height + sift_height

    frame_count = 0

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    # Grayscale of sift_image
    sift_image_gray = cv2.cvtColor(sift_image.copy(), cv2.COLOR_BGR2GRAY)

    if save_bool: # save stream
        # initalize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # MP4 Codec
        out = cv2.VideoWriter(f'{full_output_path}' + '.mp4', fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            # read new frame
            ret, frame = cap.read()
            if not ret: # break if video fails
                break
            # update data every x frames
            if frame_count % update_every_x_frame == 0:
                # grayscale of current frame
                frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
                
                # find the keypoints and descriptors with SIFT
                kp1, des1 = sift.detectAndCompute(frame_gray, None)
                kp2, des2 = sift.detectAndCompute(sift_image_gray, None)

                if rside:
                    matches = bf.knnMatch(des1, des2, k=2)
                else:
                    matches = bf.knnMatch(des2, des1, k=2)

                # Apply ratio test
                good = []
                for m, n in matches:
                    if m.distance < ratio_threshold*n.distance:
                        good.append([m])
            
            frame = sift_markup(frame, sift_image, rside, line_color, point_color, sift_flag, good, kp1, kp2)
        
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

    else: # don't save stream
        while cap.isOpened():
            # read new frame
            ret, frame = cap.read()
            if not ret: # break if video fails
                break
            # update data every x frames
            if frame_count % update_every_x_frame == 0:
                # grayscale of current frame
                frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
                
                # find the keypoints and descriptors with SIFT
                kp1, des1 = sift.detectAndCompute(frame_gray, None)
                kp2, des2 = sift.detectAndCompute(sift_image_gray, None)

                if rside:
                    matches = bf.knnMatch(des1, des2, k=2)
                else:
                    matches = bf.knnMatch(des2, des1, k=2)

                # Apply ratio test
                good = []
                for m, n in matches:
                    if m.distance < ratio_threshold*n.distance:
                        good.append([m])
            
            frame = sift_markup(frame, sift_image, rside, line_color, point_color, sift_flag, good, kp1, kp2)
            
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
