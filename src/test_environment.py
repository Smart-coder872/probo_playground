from utils import Landmark
import pytest

TEST_LANDMARKS = [
    Landmark((4, 3), 1),
    Landmark((5, 8), 2),
    Landmark((3, 2), 3),
]

def t_H_eval(lm_id):
    lm = next(l for l in TEST_LANDMARKS if l.id == lm_id)
    return lm.pos.x, lm.pos.y

def test_H_eval_landmark_2():
    result_x, result_y = t_H_eval(2)
    assert result_x == 5
    assert result_y == 8

