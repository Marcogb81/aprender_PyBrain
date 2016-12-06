# primero importamos modulos y funciones espeficas
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import  SoftmaxLayer
# importamos pylab, scipy, numpy y sus funciones
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
# importamos los modulos que sirven para construir una red neural
from pybrain.structure import FeedForwardNetwork,LinearLayer, SigmoidLayer,FullConnection
# producimos dataset con conjuento de puntos en 2d para las clases
means = [(-1,0),(2,4),(3,1)]
cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
alldata = ClassificationDataSet(2,1,nb_classes=3)
for n in xrange(3):
    for klass in range(3):
        input = multivariate_normal(means[klass],cov[klass])
        alldata.addSample(input, [klass])
# dividimos aleatoriamente el dataset en 75% entreno y 25% en conjunto de pruebas
tstdata, trndata = alldata.splitWithProportion(0.25)
# ahora clasificamos la red neuronal
trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )
# testeamos el conjunto de datos imprimiento informacion sobre el
print "Number of training patterns:", len(trndata)
print "Input and output dimension:", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0],trndata['class'][0]
# ahora construimos la red retroalimentada con cinco unidades en la capa oculta
# n = FeedForwardNetwork()
# inLayer = LinearLayer(2)
# hiddenLayer = SigmoidLayer(5)
# outLayer = LinearLayer(1)
# # ahora ensamblamos las capas
# n.addInputModule(inLayer)
# n.addModule(hiddenLayer)
# n.addOutputModule(outLayer)
# # establecemos conexiones entre capas
# in_to_hidden = FullConnection(inLayer, hiddenLayer)
# hidden_to_out = FullConnection(hiddenLayer, outLayer)
# # agregamos lass estructuras de conexion a los emsamblajes de capas
# # como veis se va muy por partes y es sistematico, lo que nos permite crear autenticos procedimientos
# # de crear redes neurales a medida estandar y atipicas, permitiendo tener una herramienta de potencia incalculable
# # e incluso dejarlo como plantilla para tener un conjunto de modulos y macro modulos que permite agurper y combinar
# n.addConnection(in_to_hidden)
# n.addConnection(hidden_to_out)
# # ahora empaquetamos todo
# n.sortModules()
# construimos red de forma rapido con todo lo anterior hecho con ela tajo buildnetwork
fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass= SoftmaxLayer)
# preparamos en entrenamiento
trainer = BackpropTrainer(fnn, dataset=trndata, momentum=0.1,verbose=True, weightdecay=0.01)
# generamos una matriz de datosy
ticks = arange(-3.,6.,0.2)
X, Y = meshgrid(ticks,ticks)
# necesitamos un vecto columan en el dataset, sin punteros
griddata = ClassificationDataSet(2,1,nb_classes=3)
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i], Y.ravel()[i]], [0])
griddata._convertToOneOfMany() # hace la red fiable
# comenzamos las iteraciones de entreno
for i in range(20):
    trainer.trainEpochs(1)
    trnresult = percentError(trainer.testOnClassData(),
                             trndata['class'])
    tstresult = percentError(trainer.testOnClassData(
        dataset=tstdata), tstdata['class'])

    print "epoch: %4d" % trainer.totalepochs, \
        "  train error: %5.2f%%" % trnresult, \
        "  test error: %5.2f%%" % tstresult
    out = fnn.activateOnDataset(griddata)
    out = out.argmax(axis=1)  # the highest output activation gives the class
    out = out.reshape(X.shape)
    figure(1)
    ioff()  # interactive graphics off
    clf()  # clear the plot
    hold(True)  # overplot on
    for c in [0, 1, 2]:
        here, _ = where(tstdata['class'] == c)
        plot(tstdata['input'][here, 0], tstdata['input'][here, 1], 'o')
    if out.max() != out.min():  # safety check against flat field
        contourf(X, Y, out)  # plot the contour
    ion()  # interactive graphics on
    draw()  # update the plot
    ioff()
    show()
