import cv2
import numpy as np


def analyze_template(image, template, method_choice, num_results=1):
    # image - image that has already been loaded in the program
    # template - template image that has already been loaded in the program. Should be smaller than image 
    # method_choice - chosen cv2 method for template matching
    # num_results - the number of top results to be displayed using bounding boxes on the returned image 

    # returns a list of tuple coordinates of rectangle to be drawn over the template match area

    # top_left = (x1, y1)
    # bottom_right = (x2, y2)
    # 
    # list.append(top_left, bottom_right) 
    # [( (x1,y1), (x2,y2)  ), (...), (...)]

    # set num_results cannot be less than 1
    if num_results < 1:
        num_results = 1

    rect_coord_list = []

    gray_image_og = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image_og = np.squeeze(gray_image_og)
    gray_image_og = gray_image_og.astype('uint8')
    
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_template = np.squeeze(gray_template)
    gray_template = gray_template.astype('uint8')

    # get width and height of image and template
    image_w, image_h = gray_image_og.shape[::-1]
    template_w, template_h = gray_template.shape[::-1]

    # throw exception if template is larger than image
    if template_w > image_w or template_h > image_h:
        raise Exception('Error: Template image larger than source image')
    
    gray_image = gray_image_og.copy()
    method = getattr(cv2, method_choice) # get enum associated with the algorithm name


    for i in range(num_results):
        # initial result
        result = cv2.matchTemplate(gray_image, gray_template, method)

        # apply template Matching
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # if the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        else:
            top_left = max_loc
            bottom_right = (top_left[0] + template_w, top_left[1] + template_h)
        cv2.rectangle(gray_image, top_left, bottom_right, 0, -1) # filled black rectangle in position of top result, so that it is not counted again 

        rect_coord = (top_left, bottom_right)
        rect_coord_list.append(rect_coord) # append set of coordinates for each match result 

    return rect_coord_list

def markup_tm(image, method_choice, top_left, bottom_right, box_color, text_color, text_size, text_thick, box_thick, outline_bool, outline_color):
    # marks the method_choice in the top left corner of the image
    # creates a bounding box on the image over the coordinates provided by analyze_template

    # draw bounding box for template match
    cv2.rectangle(image, top_left, bottom_right, box_color, box_thick) # draw result rectangle

    # text
    text_corner = [0, 60]
    text_corner[1] = int(text_corner[1] * text_size)
    if outline_bool:
        image = cv2.putText(image, f'TM method: {method_choice}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

    image = cv2.putText(image, f'TM method: {method_choice}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color, thickness=text_thick)


def template_match(img, template, methods, box_color, text_color, text_size, text_thick, box_thick, outline_bool, outline_color, full_output_path, method_choice=None, num_results=1):

    paths_list = []
    img_list = []

    # top_left = (x1, y1)
    # bottom_right = (x2, y2)
    # 
    # rect_coord_list.append(top_left, bottom_right) 
    # [( (x1,y1), (x2,y2)  ), (...), (...)]

    if method_choice in methods:
        image = img.copy()
        # use one method specified by method_choice. 
        rect_coord_list = analyze_template(image, template, method_choice, num_results)

        for j in range(len(rect_coord_list)):
            markup_tm(image, method_choice, rect_coord_list[j][0], rect_coord_list[j][1], box_color, text_color, text_size, text_thick, box_thick, outline_bool, outline_color)

        # write output image to .jpg and place in a folder
        path = f'{full_output_path}_{method_choice}.jpg'
        paths_list.append(path), img_list.append(image)

    else:
        # use all 6 methods
        for i in range(len(methods)):
            image = img.copy()
            method_choice = methods[i]
            rect_coord_list = analyze_template(image, template, method_choice, num_results)

            for j in range(len(rect_coord_list)):
                markup_tm(image, method_choice, rect_coord_list[j][0], rect_coord_list[j][1], box_color, text_color, text_size, text_thick, box_thick, outline_bool, outline_color)

            # write output image to .jpg and place in a folder
            path = f'{full_output_path}_{method_choice}.jpg'
            paths_list.append(path), img_list.append(image)

    
    return paths_list, img_list

def template_match_video(video_path, update_every_x_frame, template, methods, box_color, text_color, text_size, text_thick, box_thick, outline_bool, outline_color, full_output_path, text_color_methods, method_choice=None):
    # update_every_x_frame -Hhow often template match data is updated. If a face is detected on the frame, the bounding box will stay in the same position in the video until face data is updated again

    # Read the video 
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))     # Width of frame
    frame_height = int(cap.get(4))    # Height of frame
    fps = cap.get(cv2.CAP_PROP_FPS)   # Frames per second

    # Initalize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # MP4 Codec
    if method_choice not in methods:
        out = cv2.VideoWriter(f'{full_output_path}.mp4', fourcc, fps, (frame_width, frame_height))
    else:
        out = cv2.VideoWriter(f'{full_output_path}_{method_choice}.mp4', fourcc, fps, (frame_width, frame_height))

   
    frame_count = 0
    if method_choice not in methods:
        while cap.isOpened():
            # Read new frame
            ret, frame = cap.read()
            if not ret: # break if video fails
                break

            # starting at zero, then every x frames update rectangles position data
            if frame_count % update_every_x_frame == 0:  
                rect_coord_list = []
                for i in range(len(methods)):
                    temp = analyze_template(frame, template, methods[i], num_results=1)
                    rect_coord_list = rect_coord_list + temp
                
            # draw rectangles and corresponding text every frame
            for x in range(len(rect_coord_list)):
                cv2.rectangle(frame, rect_coord_list[x][0], rect_coord_list[x][1], text_color_methods[x], box_thick)
                
                if outline_bool:
                    frame = cv2.putText(frame, methods[x], org=(rect_coord_list[x][0][0], rect_coord_list[x][0][1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

                frame = cv2.putText(frame, methods[x], org=(rect_coord_list[x][0][0], rect_coord_list[x][0][1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color_methods[x], thickness=text_thick)

            out.write(frame)
            frame_count += 1
    else:
        while cap.isOpened():
            # Read new frame
            ret, frame = cap.read()
            if not ret: # break if video fails
                break

            # method name labeled in the corner for every frame
            text_corner = [0, 60]
            text_corner[1] = int(text_corner[1] * text_size)
            if outline_bool:
                frame = cv2.putText(frame, f'TM method: {method_choice}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

            frame = cv2.putText(frame, f'TM method: {method_choice}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color, thickness=text_thick)

        
            # starting at zero, then every x frames update rectangles position data
            if frame_count % update_every_x_frame == 0:  
                rect_coord_list = analyze_template(frame, template, method_choice, num_results=1)
            
            # draw rectangles every frame
            for x in range(len(rect_coord_list)):
                cv2.rectangle(frame, rect_coord_list[x][0], rect_coord_list[x][1], box_color, box_thick)

            out.write(frame)
            frame_count += 1

    # Release everything if job is finished
    cap.release()
    out.release()

def template_match_stream(camera_index, endkey, save_bool, update_every_x_frame, template, methods, box_color, text_color, text_size, text_thick, box_thick, outline_bool, outline_color, full_output_path, text_color_methods, method_choice=None):
    # read the video 
    cap = cv2.VideoCapture(camera_index)
    frame_width = int(cap.get(3))     # Width of frame
    frame_height = int(cap.get(4))    # Height of frame
    fps = cap.get(cv2.CAP_PROP_FPS)   # Frames per second

    frame_count = 0

    if save_bool: # save livestream
        # initalize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # MP4 Codec
        if method_choice not in methods:
            out = cv2.VideoWriter(f'{full_output_path}.mp4', fourcc, fps, (frame_width, frame_height))
        else:
            out = cv2.VideoWriter(f'{full_output_path}_{method_choice}.mp4', fourcc, fps, (frame_width, frame_height))
   
        if method_choice not in methods: # multi method
            while cap.isOpened():
                # Read new frame
                ret, frame = cap.read()
                if not ret: # break if video fails
                    break

                # starting at zero, then every x frames update rectangles position data
                if frame_count % update_every_x_frame == 0:  
                    rect_coord_list = []
                    for i in range(len(methods)):
                        temp = analyze_template(frame, template, methods[i], num_results=1)
                        rect_coord_list = rect_coord_list + temp
                    
                # draw rectangles and corresponding text every frame
                for x in range(len(rect_coord_list)):
                    cv2.rectangle(frame, rect_coord_list[x][0], rect_coord_list[x][1], text_color_methods[x], box_thick)
                    
                    if outline_bool:
                        frame = cv2.putText(frame, methods[x], org=(rect_coord_list[x][0][0], rect_coord_list[x][0][1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

                    frame = cv2.putText(frame, methods[x], org=(rect_coord_list[x][0][0], rect_coord_list[x][0][1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color_methods[x], thickness=text_thick)

                out.write(frame)
                frame_count += 1
                   
                cv2.namedWindow('Live Stream', cv2.WINDOW_NORMAL)
                cv2.imshow('Live Stream', frame)
                # Check if 'X' button is clicked or endkey is pressed
                key = cv2.waitKey(1) & 0xFF  # waitKey(1) for continuous frame updates
                if key == ord(endkey) or cv2.getWindowProperty('Live Stream', cv2.WND_PROP_VISIBLE) < 1:
                    break

        else: # single method
            while cap.isOpened(): 
                # Read new frame
                ret, frame = cap.read()
                if not ret: # break if video fails
                    break

                # method name labeled in the corner for every frame
                text_corner = [0, 60]
                text_corner[1] = int(text_corner[1] * text_size)
                if outline_bool:
                    frame = cv2.putText(frame, f'TM method: {method_choice}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

                frame = cv2.putText(frame, f'TM method: {method_choice}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color, thickness=text_thick)
            
                # starting at zero, then every x frames update rectangles position data
                if frame_count % update_every_x_frame == 0:  
                    rect_coord_list = analyze_template(frame, template, method_choice, num_results=1)
                
                # draw rectangles every frame
                for x in range(len(rect_coord_list)):
                    cv2.rectangle(frame, rect_coord_list[x][0], rect_coord_list[x][1], box_color, box_thick)

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

    else: # don't save livestream
        if method_choice not in methods: # multi method
            while cap.isOpened():
                # Read new frame
                ret, frame = cap.read()
                if not ret: # break if video fails
                    break

                # starting at zero, then every x frames update rectangles position data
                if frame_count % update_every_x_frame == 0:  
                    rect_coord_list = []
                    for i in range(len(methods)):
                        temp = analyze_template(frame, template, methods[i], num_results=1)
                        rect_coord_list = rect_coord_list + temp
                    
                # draw rectangles and corresponding text every frame
                for x in range(len(rect_coord_list)):
                    cv2.rectangle(frame, rect_coord_list[x][0], rect_coord_list[x][1], text_color_methods[x], box_thick)
                    
                    if outline_bool:
                        frame = cv2.putText(frame, methods[x], org=(rect_coord_list[x][0][0], rect_coord_list[x][0][1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

                    frame = cv2.putText(frame, methods[x], org=(rect_coord_list[x][0][0], rect_coord_list[x][0][1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color_methods[x], thickness=text_thick)

                frame_count += 1
                   
                cv2.namedWindow('Live Stream', cv2.WINDOW_NORMAL)
                cv2.imshow('Live Stream', frame)
                # Check if 'X' button is clicked or endkey is pressed
                key = cv2.waitKey(1) & 0xFF  # waitKey(1) for continuous frame updates
                if key == ord(endkey) or cv2.getWindowProperty('Live Stream', cv2.WND_PROP_VISIBLE) < 1:
                    break

        else: # single method
            while cap.isOpened(): 
                # Read new frame
                ret, frame = cap.read()
                if not ret: # break if video fails
                    break

                # method name labeled in the corner for every frame
                text_corner = [0, 60]
                text_corner[1] = int(text_corner[1] * text_size)
                if outline_bool:
                    frame = cv2.putText(frame, f'TM method: {method_choice}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=outline_color, thickness=text_thick*4)

                frame = cv2.putText(frame, f'TM method: {method_choice}', org=text_corner, fontFace=cv2. FONT_HERSHEY_SIMPLEX, fontScale=text_size, color=text_color, thickness=text_thick)
            
                # starting at zero, then every x frames update rectangles position data
                if frame_count % update_every_x_frame == 0:  
                    rect_coord_list = analyze_template(frame, template, method_choice, num_results=1)
                
                # draw rectangles every frame
                for x in range(len(rect_coord_list)):
                    cv2.rectangle(frame, rect_coord_list[x][0], rect_coord_list[x][1], box_color, box_thick)

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



    