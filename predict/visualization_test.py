# encoding:utf-8

import visdom

vis = visdom.Visdom(env='model_2')

vis.text('Hello World', win='text1')