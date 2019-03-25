def c(t, value):
    if t > 0:
        value = (value+3000)*1.01
        c(t-1,value)
    else:
        print(value)


c(12,3000)