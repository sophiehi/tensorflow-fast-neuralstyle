import tensorflow as tf

print "tfv:", tf.__version__
print "hello pypyp"

class MyClass():
    def __init__(self):
        print "init()"

    def __call__(self, a):
        print "call()", a

c = MyClass()
c("aaa")
