
import pytest
import numpy as np


from environment import Environment
from numpy import sqrt, atan2
from utils import Landmark
from numpy import sqrt, atan2
from pandas import DataFrame

class TestEnv:
    @pytest.fixture(autouse=True)
    def __init__(self, test_landmarks):
        test_landmarks = [Landmark((4,3), 1),
                          Landmark((5,8), 2),
                          Landmark((3,2), 3)]
        self.TEST_LANDMARKS = test_landmarks

    def t_H_eval(self, lm_id):
        # find the matching landmark object
        lm = next(l for l in self.TEST_LANDMARKS if l.id == lm_id)

        # extract its position
        lm_x = lm.pos.x
        lm_y = lm.pos.y

        # ...use lm_x, lm_y plus state x to build H or h(x,lm)...
        # for now, just return them:
        return lm_x, lm_y
    def test_H_eval_landmark_2(self):
        result_x, result_y = self.t_H_eval(2)  # Now call the helper method
        assert result_x == 5
        assert result_y == 8

