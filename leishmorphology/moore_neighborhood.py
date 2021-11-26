import numpy as np

class MaxIterReachedError(Exception):
    def __str__(self):
        return "Reached max_iter, something likely went wrong with the stopping condition."

clockwise = [(-1,0),(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1)]
M = {
    (-1,1): 0,
    (0,1): 7,
    (1,1): 6,
    (-1,0): 1,
    (1,0): 5,
    (-1,-1): 2,
    (0,-1): 3,
    (1,-1): 4,
}
def moore_neighborhood(mask, max_iter=1e4):
    B = []
    b = (-1,-1)
    s = (-1,-1)
    break_cond = False
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            b = s
            s = (i,j)
            if mask[s]:
                break_cond = True
                break
        if break_cond:
            break
    B.append(s)

    p = s
    Δ = (s[0]-b[0], s[1]-b[1])
    clockwise_index = M[Δ]
    shift = clockwise[clockwise_index]
    c = (s[0]+shift[0], s[1]+shift[1])
    while c != s:
        if mask[c]:
            B.append(c)
            b = p
            p = c
            Δ = (p[0]-b[0], p[1]-b[1])
            clockwise_index = M[Δ]
            shift = clockwise[clockwise_index]
        else:
            clockwise_index -= 1
            if clockwise_index < 0:
                clockwise_index = 7
            shift = clockwise[clockwise_index]
        c = (p[0]+shift[0], p[1]+shift[1])
        if len(B)>max_iter:
            raise MaxIterReachedError()
    return np.array(B)
