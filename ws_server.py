#!/usr/bin/python

###############################################################################
# MIT License (MIT)
#
# Copyright (c)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
###############################################################################

from autobahn.twisted.websocket import WebSocketServerProtocol, \
    WebSocketServerFactory
from json import loads as json_loads
#from dispatcher import *
from random import uniform
from sys import stdout
from twisted.python import log
from twisted.internet import reactor
from gc import collect as gc_collect
from json import dumps as json_dumps

DEBUG = True
PORT = 8000
URL = "ws://8.8.8.8:%d"
MAX_CON = 100

"""@package docstring
Websocket interface module.

Websocket interface implementation for prediction model implementation, input
interfaces are: studentID, and a list of courseID, provides the quality and
prediction values after classification/estimation process.
"""

class BackendServerProtocol(WebSocketServerProtocol):
    dispatchers = {}

    def onConnect(self, request):
        print("Client connecting: %s"%( request.peer ))

    def onOpen(self):
        print("WebSocket connection open.")

    def onMessage(self, payload, isBinary):
        #dispatcher = WSDispatcher()
        if isBinary:
            print("Binary message received: %d bytes"%( len( payload ) ))
        else:
            json_string = format(payload.decode('utf8'))
            json_input = json_loads( json_string )
            #pprint( json_input )
        #try:
        #    dispatcher = self.dispatchers[self.peer]
        #except:
        #    dispatcher = WSDispatcher()
        #    self.dispatchers[self.peer] = dispatcher

        # echo back message verbatim
        #self.sendMessage( dispatcher.risk( json_input ), False )
        result = {'risk':uniform(0,1),'quality':uniform(0,1)}
        self.sendMessage( json_dumps(result), False )
        
    def onClose(self, wasClean, code, reason):
        #try:
        #    del(self.dispatchers[self.peer])
        #except:
        #    pass        
        gc_collect()
        print("WebSocket connection closed: %s"%( reason ))

if __name__ == '__main__':

    log.startLogging( stdout )

    factory = WebSocketServerFactory(URL%PORT, debug=DEBUG)
    factory.protocol = BackendServerProtocol
    factory.setProtocolOptions(maxConnections=MAX_CON)

    reactor.listenTCP( PORT, factory )
    reactor.run()
