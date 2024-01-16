# class for type 4 (i.e. Dominic's) grid

import math
import pandas as pd


class Grid:
    # bounding_box -> bounding box of an object
    # type -> type of an object (i.e. feeder, cup, crate)
    # configr -> configuration of assembly task (tuple)
    def __init__(self):
        self.objsPlcd = []
        self.shift
        
    # TODO: convert to 3D
    def compute_quadrant(self, bounding_box, type, configr):
        object_center_x = ((bounding_box[2][0] - bounding_box[0][0]) / 2) 
        object_center_y = ((bounding_box[2][1] - bounding_box[0][1]) / 2)
        config_placements = [] # coordinates of the configuration

        if (len(self.objsPlcd) == 0): # if no objs yet on grid
            self.shift = []
            for obj_tup in configr: # compute shift needed to create necessary coordinates for all objects of the same type (e.g. all cups)
                if (type == obj_tup[0]):
                    config_placements.append((obj_tup[1][0], obj_tup[1][1]))
                    self.shift.append((object_center_x-obj_tup[1][0], object_center_y-obj_tup[1][1]))
            
        else: # if first object was already placed on grid  
            for shift_tup in self.shift: # if there are multiple shifts because of multiple possiibilities then check all
                estimated_pos = (object_center_x-shift_tup[0], object_center_y-shift_tup[1])
                
                for obj_tup in configr: # check for the closest viable position 
                    distances = []
                    possibilities = []
                    if (type == obj_tup[0]):
                        possibilities.append((obj_tup[1][0], obj_tup[1][1]))
                        distances.append(math.sqrt(pow(obj_tup[1][0]-estimated_pos[0]) + pow(obj_tup[1][1]-estimated_pos[1])))
                    
                config_placements.append(possibilities[pd.Series(distances).idxmin()]) # get the closest viable position in the configuration coordinate system       
        
        self.objsPlcd.append(type)
        return config_placements