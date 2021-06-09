import cv2
import numpy as np

# all possible matching methods
all_methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

best_methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED'] # highest accuracy methods during inital testing


def templateMatching(champ_img, map_img, methods=all_methods):
    """ attampts to match a template image within a larger image.
    Params
    ---------
        champ_img : str
            file path to champion template image
        map_img : str
            file path to map image
        methods: list
            list of matching methods to use (defaults to all  methods)
    """
    # read images
    map_img = cv2.imread(map_img, 0)
    template = cv2.resize(cv2.imread(champ_img, 0),(27,27))
    width, height = template.shape
    
    for m in methods:
        method = eval(m)
        map_copy = map_img.copy()
        res = cv2.matchTemplate(map_copy, template, method) # Apply template Matching
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + width, top_left[1] + height)

        cv2.rectangle(map_copy, top_left, bottom_right, 255, 2) # draw rect around match
        cv2.imshow(f'Match using {m}', map_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':

    map_img_path = 'assets/mini map/'
    champ_img_path = 'assets/champions/ezreal.png'

    for i in range(1,6): # iterate maps
        templateMatching(champ_img_path, f'{map_img_path}map{i}.png', best_methods)


