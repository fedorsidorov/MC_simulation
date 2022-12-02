from matplotlib import rc
rc('font', **{'family':'serif'})
rc('text', usetex=True)

rc('text.latex', unicode=True)
rc('text.latex', preamble=r'\usepackage[utf8]{inputenc}')
rc('text.latex', preamble=r'\usepackage[russian]{babel}')

from pylab import *


def figsize(wcm,hcm):
    figure(figsize=(wcm/2.54,hcm/2.54))
    figsize(13,9)


x = linspace(0,2*pi,100)
y = sin(x)
plot(x,y,'-')
xlabel(u"ось абсцисс")
ylabel(u"ось ординат")
title(u"Две беды в России — синусы и косинусы!")