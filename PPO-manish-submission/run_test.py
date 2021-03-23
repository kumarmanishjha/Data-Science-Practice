from challenge import SpecPredict
import numpy
print('Manish')

model = SpecPredict('model.pth')
x1 = numpy.random.rand(1,502)
print('Single Input Sample:' +str(x1))
print('Predicted output')
print(model.predict(x1))
print('------------------')
x2 = numpy.random.rand(10,502)
print('Batch Input data:' +str(x2))
print('Predicted output')
print(model.predict(x2))