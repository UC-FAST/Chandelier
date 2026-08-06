def b():
    a=yield
    print(a)


if __name__=='__main__':
    f=b()
    next(f)
    f.send(1)