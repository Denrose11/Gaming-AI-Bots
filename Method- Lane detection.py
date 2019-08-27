import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
from numpy import ones,vstack
from numpy.linalg import lstsq
from directkeys import PressKey,ReleaseKey, W, A, S, D #taken from:- http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
from statistics import mean

def Region_of_interest(Image, vertices_we_focused_on):
    
    #Making here(at first it will be blank)
    mask_we_focused_on = np.zeros_like(Image)
    #We will fill the pixels inside the polygon whihc are defined by the vertices_we_focused_on
    cv2.fillPoly(mask_we_focused_on, vertices_we_focused_on, 255)
    
    #we will return the image where the masking is now used
    mask_we_made = cv2.bitwise_and(Image, mask_we_focused_on)
    return mask_we_made


def lanes_we_draw(Image, lines_formed, colours=[0, 255, 255], thick=3):

    #If this method fails, the default lines will be used
    try:

        #Finding the maximum value of y to mark the lane
        # (because we cannot presume that the horizon will at the same point)

        y_corr = []  
        for i in lines_formed:
            for ii in i:
                y_corr += [ii[1],ii[3]]
        minimum_of_y = min(y_corr)# we draw lines to this point evertime
        maximum_of_y = 600 #  thats the maximum y point we can have
        newly_formed_lines = []
        dictionary_of_lines = {}

        for idx,i in enumerate(lines_formed):
            for x_points in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # It is used to get the lines in the set given two set of coordinates.
                corrdinates_of_x = (x_points[0],x_points[2]) # calculating the definition of the line
                corrdinates_of_y = (x_points[1],x_points[3])
                A = vstack([corrdinates_of_x,ones(len(corrdinates_of_x))]).T
                m, b = lstsq(A, corrdinates_of_y)[0]

                # Calculating our improved xs
                x1 = (minimum_of_y-b) / m
                x2 = (maximum_of_y-b) / m

                dictionary_of_lines[idx] = [m,b,[int(x1), minimum_of_y, int(x2), maximum_of_y]] #stores the slope bais and actual y and x values
                newly_formed_lines.append([int(x1), minimum_of_y, int(x2), maximum_of_y])

        final_lanes = {}

        for idx in dictionary_of_lines:
            final_lanes_copy = final_lanes.copy()
            m = dictionary_of_lines[idx][0] # we are checking fo the slope
            b = dictionary_of_lines[idx][1] # we are basically trying to find the lines which have similar slopes
            line = dictionary_of_lines[idx][2]
            
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8): #slope
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8): #bias
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ] # took two most common slopes m ,b

        counter_of_lines = {}

        for lanes_1 in final_lanes:
            counter_of_lines[lanes_1] = len(final_lanes[lanes_1])

        level_top_lanes = sorted(counter_of_lines.items(), key=lambda item: item[1])[::-1][:2]

        id_lane1 = level_top_lanes[0][0]
        id_lane2 = level_top_lanes[1][0]

        def lane_average(data_lane):
            meas_x1 = []
            meas_y1 = []
            meas_x2 = []
            meas_y2 = []
            for data in data_lane:
                meas_x1.append(data[2][0])
                meas_y1.append(data[2][1])
                meas_x2.append(data[2][2])
                meas_y2.append(data[2][3])
            return int(mean(meas_x1)), int(mean(meas_y1)), int(mean(meas_x2)), int(mean(meas_y2)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = lane_average(final_lanes[id_lane1])
        l2_x1, l2_y1, l2_x2, l2_y2 = lane_average(final_lanes[id_lane2])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2], id_lane1, id_lane2
    except Exception as e:
        print(str(e))


def image_process(image):
    original_image = image
    # used for edge detection
    image_processed =  cv2.Canny(image, threshold1 = 200, threshold2=300)# tweakable
    
    image_processed = cv2.GaussianBlur(image_processed,(5,5),0) #blurred using gaussain blur to touch the sperated lines.
    
    vertices_we_focused_on = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500],
                         ], np.int32) #suitable for scooter only

    image_processed = Region_of_interest(image_processed, [vertices_we_focused_on])

    # more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    #                                     rho   theta   thresh  min length, max gap:        
    lines_formed = cv2.HoughLinesP(image_processed, 1, np.pi/180, 180,      20,       15)
    m1 = 0 #defaults for lines
    m2 = 0
    try:
        l1, l2, m1,m2 = lanes_we_draw(original_image,lines_formed)
        cv2.line(original_image, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(original_image, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
    except Exception as e:
        print(str(e))
        pass
    try:
        for coords in lines_formed:
            coords = coords[0]
            try:
                cv2.line(image_processed, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                
                
            except Exception as e:
                print(str(e))
    except Exception as e:
        pass

    return image_processed,original_image, m1, m2

def straight(): # simple driving directions defined
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(A) # just a little left

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

def slowing_down():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)


last_time = time.time()
while True:
    screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
    print('Frame took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    new_screen,original_image, m1, m2 = image_process(screen) #m1,m2 are slopes
    cv2.imshow('window2',cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))



    if m1 < 0 and m2 < 0: # both slopes are negative
        right()
    elif m1 > 0  and m2 > 0: #both slopes are positive
        left()
    else:
        straight()
    

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break



