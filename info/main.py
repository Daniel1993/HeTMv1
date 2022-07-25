def time(R, D, B):
    S = 0.0
    for i in range(1,R-1):
        M = 1.0
        for j in range(1,R-1):
            M *= 1.0/j
        S += M
    return S
    
print(time(3, 10, 1))
