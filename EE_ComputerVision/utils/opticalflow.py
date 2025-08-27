import cv2
import numpy as np

def optical_flow(img1, img2, full_output_path, total_random_colors=1000, color=None):

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
    # Select good points (st==1 where corresponding feature was found)
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
    path = f'{full_output_path}.jpg'
    return path, img



def optical_flow_video(video_path, full_output_path, total_random_colors=1000, color=None):
    # Read the video 
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))     # Width of frame
    frame_height = int(cap.get(4))    # Height of frame
    fps = cap.get(cv2.CAP_PROP_FPS)   # Frames per second

    # Initalize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # MP4 Codec
    out = cv2.VideoWriter(f'{full_output_path}' + '.mp4', fourcc, fps, (frame_width, frame_height))

    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
 
    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
 
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # reset back to first frame before entering loop
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
 
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    if color is None:
        # If color not specified, use random colors
        color_random = np.random.randint(0, 255, (total_random_colors, 3))
        while cap.isOpened():
            # Read new frame
            ret, frame = cap.read()
            if not ret: # break if video fails
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # Calculate Optical Flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        
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
                frame = cv2.circle(frame, (a, b), 5, color_random[j].tolist(), -1)
        
            # Save frame
            img = cv2.add(frame, mask)
            out.write(img)        
        
            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            # no features detected in the previous frame
            if p0 is None or len(p0) == 0:
                # set the features to the current frame
                # reset mask so that it does not fill up the video with a mess of colors 
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                mask = np.zeros_like(old_frame)
    else:
        while cap.isOpened():
            # Read new frame
            ret, frame = cap.read()
            if not ret: # break if video fails
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            # Calculate Optical Flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
        
            # Draw the tracks
            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                a = int(a)
                b = int(b)
                c = int(c)
                d = int(d)
                mask = cv2.line(mask, (a, b), (c, d), color, 2)
                frame = cv2.circle(frame, (a, b), 5, color, -1)
        
            # Save frame
            img = cv2.add(frame, mask)
            out.write(img)        
        
            # Update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

            # no features detected in the previous frame
            if p0 is None or len(p0) == 0:
                # set the features to the current frame
                # reset mask so that it does not fill up the video with a mess of colors 
                p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                mask = np.zeros_like(old_frame)

    # Release everything if job is finished
    cap.release()
    out.release()

def optical_flow_stream(camera_index, endkey, save_bool, full_output_path, total_random_colors=1000, color=None):   
    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Parameters for Lucas Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    # Read the video 
    cap = cv2.VideoCapture(camera_index)
    frame_width = int(cap.get(3))     # Width of frame
    frame_height = int(cap.get(4))    # Height of frame
    fps = cap.get(cv2.CAP_PROP_FPS)   # Frames per second

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # reset back to first frame before entering loop
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    if save_bool: # save livestream
        # Initalize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # MP4 Codec
        out = cv2.VideoWriter(f'{full_output_path}' + '.mp4', fourcc, fps, (frame_width, frame_height))

        if color is None: # if color not specified, use random colors
            color_random = np.random.randint(0, 255, (total_random_colors, 3))
            while cap.isOpened():
                # Read new frame
                ret, frame = cap.read()
                if not ret: # break if video fails
                    break

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
                # Calculate Optical Flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, p0, None, **lk_params
                )
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            
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
                    frame = cv2.circle(frame, (a, b), 5, color_random[j].tolist(), -1)
            
                # Save frame
                img = cv2.add(frame, mask)
                out.write(img)        
            
                # Update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

                # no features detected in the previous frame
                if p0 is None or len(p0) == 0:
                    # set the features to the current frame
                    # reset mask so that it does not fill up the video with a mess of colors 
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                    mask = np.zeros_like(old_frame)

                cv2.namedWindow('Live Stream', cv2.WINDOW_NORMAL)
                cv2.imshow('Live Stream', img)
                # Check if 'X' button is clicked or endkey is pressed
                key = cv2.waitKey(1) & 0xFF  # waitKey(1) for continuous frame updates
                if key == ord(endkey) or cv2.getWindowProperty('Live Stream', cv2.WND_PROP_VISIBLE) < 1:
                    break

        else: # specific color
            while cap.isOpened():
                # Read new frame
                ret, frame = cap.read()
                if not ret: # break if video fails
                    break

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
                # Calculate Optical Flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, p0, None, **lk_params
                )
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            
                # Draw the tracks
                for j, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a = int(a)
                    b = int(b)
                    c = int(c)
                    d = int(d)
                    mask = cv2.line(mask, (a, b), (c, d), color, 2)
                    frame = cv2.circle(frame, (a, b), 5, color, -1)
            
                # Save frame
                img = cv2.add(frame, mask)
                out.write(img)        
            
                # Update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

                # no features detected in the previous frame
                if p0 is None or len(p0) == 0:
                    # set the features to the current frame
                    # reset mask so that it does not fill up the video with a mess of colors 
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                    mask = np.zeros_like(old_frame)

                cv2.namedWindow('Live Stream', cv2.WINDOW_NORMAL)
                cv2.imshow('Live Stream', img)
                # Check if 'X' button is clicked or endkey is pressed
                key = cv2.waitKey(1) & 0xFF  # waitKey(1) for continuous frame updates
                if key == ord(endkey) or cv2.getWindowProperty('Live Stream', cv2.WND_PROP_VISIBLE) < 1:
                    break

        # Release everything if job is finished
        cv2.destroyAllWindows()
        cap.release()
        out.release()

    else: # don't save livestream
        if color is None: # if color not specified, use random colors
            color_random = np.random.randint(0, 255, (total_random_colors, 3))
            while cap.isOpened():
                # Read new frame
                ret, frame = cap.read()
                if not ret: # break if video fails
                    break

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
                # Calculate Optical Flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, p0, None, **lk_params
                )
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            
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
                    frame = cv2.circle(frame, (a, b), 5, color_random[j].tolist(), -1)
            
                # Save frame
                img = cv2.add(frame, mask)
            
                # Update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

                # no features detected in the previous frame
                if p0 is None or len(p0) == 0:
                    # set the features to the current frame
                    # reset mask so that it does not fill up the video with a mess of colors 
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                    mask = np.zeros_like(old_frame)

                cv2.namedWindow('Live Stream', cv2.WINDOW_NORMAL)
                cv2.imshow('Live Stream', img)
                # Check if 'X' button is clicked or endkey is pressed
                key = cv2.waitKey(1) & 0xFF  # waitKey(1) for continuous frame updates
                if key == ord(endkey) or cv2.getWindowProperty('Live Stream', cv2.WND_PROP_VISIBLE) < 1:
                    break

        else: # specific color
            while cap.isOpened():
                # Read new frame
                ret, frame = cap.read()
                if not ret: # break if video fails
                    break

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
                # Calculate Optical Flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray, p0, None, **lk_params
                )
                # Select good points
                good_new = p1[st == 1]
                good_old = p0[st == 1]
            
                # Draw the tracks
                for j, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a = int(a)
                    b = int(b)
                    c = int(c)
                    d = int(d)
                    mask = cv2.line(mask, (a, b), (c, d), color, 2)
                    frame = cv2.circle(frame, (a, b), 5, color, -1)
            
                # Save frame
                img = cv2.add(frame, mask)
            
                # Update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)

                # no features detected in the previous frame
                if p0 is None or len(p0) == 0:
                    # set the features to the current frame
                    # reset mask so that it does not fill up the video with a mess of colors 
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                    mask = np.zeros_like(old_frame)

                cv2.namedWindow('Live Stream', cv2.WINDOW_NORMAL)
                cv2.imshow('Live Stream', img)
                # Check if 'X' button is clicked or endkey is pressed
                key = cv2.waitKey(1) & 0xFF  # waitKey(1) for continuous frame updates
                if key == ord(endkey) or cv2.getWindowProperty('Live Stream', cv2.WND_PROP_VISIBLE) < 1:
                    break

        # Release everything if job is finished
        cv2.destroyAllWindows()
        cap.release()
