from PIL import ImageGrab
import numpy as np
import cv2
from image_matching import *
import time

w, h = 2560, 1440 # screen res
map_size = 360

best_methods = ['cv2.TM_CCOEFF']
champ_img_path = 'assets/champions/ezreal.png'

template = cv2.resize(cv2.imread(champ_img_path, 0),(27,27))
width, height = template.shape

while True:
    screen = np.array(ImageGrab.grab(bbox=(w-map_size, h-map_size, w-10, h-10))) # get mini map image
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) # change to grey scale
    screen_copy = screen_gray.copy()

    # Apply template Matching
    res = cv2.matchTemplate(screen_copy, template, cv2.TM_CCOEFF) 
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)

    cv2.rectangle(screen_copy, top_left, bottom_right, 255, 2) # draw rect around match
    cv2.imshow('Matching with cv2.TM_CCOEFF', screen_copy)

    if cv2.waitKey(25) & 0xFF == ord('q'): # run until user quits
        cv2.destroyAllWindows()
        break
