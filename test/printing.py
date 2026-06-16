
import numpy as np
from src.utils import Position, Pose, Bounds, Landmark, BearingRange, wrap_angle

posList = [
        Landmark(Position(2.0, 2.0), id=0),
        Landmark(Position(5.0, 5.0), id=1),
        Landmark(Position(8.0, 8.0), id=2),
    ]

for landmark in posList:
    print(posList[landmark][0])