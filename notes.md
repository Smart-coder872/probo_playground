# Notes I created to reference
## utils.py 
- Position class      --> returns xy robot cordinates
    - Objects: 'x' and 'y'
- Pose class          --> returns Position with direction
    - Objects: 'theta' and 'pos:Position = Position()    
- Bounds class        --> checks if an xy corrdinate is within bounds
    - Objects: 'x_min', 'x_max', 'y_min' and 'y_max'
    - func: 'within_x (x)', 'within_y (y)' and 'within_bounds (pos)
- Landmark class      --> identifiable naviagation aid
    - Objects: 'pos' and 'id'
- Bearing Range class --> landmark and robot relationship
    - Objects: 'landmark_id', 'bearing', 'range'
## __init__
- dimensions: uses Bounds
- dt: a float number
- obstacles: uses Bounds listed
- landmarks: uses Landmark listed
- robot_starting_pose: uses Pose

## Storage
- self.robot pose = robot_starting_pose = Pose = theta + pos:Position = Position()
- self.DIMENSIONS = dimensions = Bounds


## Questions
- In the utils.py file, what is the difference between the object 'id' in the 'Landmark' class and the object 'landmark_id' in the 'Bearing Range' class?
- In the utils.py file, what is bearing and what does the object 'bearing' in the "Bearing range" class represent?

