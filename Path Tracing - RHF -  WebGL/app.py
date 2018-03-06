from gevent import monkey
monkey.patch_all()

import cgi
import redis
from flask import Flask, render_template, request
from flask_socketio import SocketIO

import base64
import OpenEXR
import time
import sys, os
import png


app = Flask(__name__)
db = redis.StrictRedis('localhost', 6379, 0)
socketio = SocketIO(app)


# sudo apt-get install redis-server
# pip install -r requirements.txt

@app.route('/')
def main():
    return render_template('pathTracingDemo.html')



@socketio.on('connect')
def ws_conn():
    c = db.incr('connected')
    socketio.emit('msg', {'count': c})

# When this user emits, client side: socket.emit('otherevent',some data);
@socketio.on('mouse')
def get_data(json):
    # Data comes in as whatever was sent, including objects
    # print("Received: 'mouse' " + str(data.x) + " " + str(data.y));
    print("Received: 'mouse' ");
    # print(json);

    start = time.time()

    image_64_decode = base64.decodestring(json["channel3"]) 

    print(type(image_64_decode))
    image_result = open('deer_decode3.png', 'wb') # create a writable image and write the decoding result
    image_result.write(image_64_decode)

    end = time.time()
    print("Elapsed time")
    print(end - start)


@socketio.on('image')
def get_data(json):

    start = time.time()
    image_64_decode = base64.decodestring(json["image"]) 

    print(type(image_64_decode))
    image_result = open('static/rhf/image.png', 'wb') # create a writable image and write the decoding result
    image_result.write(image_64_decode)

    end = time.time()
    print("Elapsed time on saving image")
    print(end - start)

    os.system("rhfCpp/main.out foo.exr -h bar.exr -d 0.8 static/rhf/image_filt.exr")

    # >>> import OpenEXR
    # >>> golden = OpenEXR.InputFile("GoldenGate.exr")
    # >>> (r, g, b) = golden.channels("RGB")
    # >>> print len(r), len(g), len(b)
    # 2170640 2170640 2170640
    # os.system("bash scripts.sh")

    # os.system("rhfCpp/main.out foo.exr -h bar.exr -d 0.8 static/rhf/image_filt.exr && rhfCpp/exrtopng static/rhf/image_filt.exr  static/rhf/image_filt.png")
    # os.system("rhfCpp/exrtopng static/rhf/image_filt.exr  static/rhf/image_filt.png")

    


# When this user emits, client side: socket.emit('otherevent',some data);
@socketio.on('histogram')
def get_data(json):

    start = time.time()
    for sample in xrange(1, json["samples"] +1):
        image_64_decode = base64.decodestring(json["channel"+ str(sample)]) 
        image_result = open('static/rhf/sample' + str(sample) + '.png', 'wb') # create a writable image and write the decoding result
        image_result.write(image_64_decode)

    end = time.time()
    print("Elapsed time on saving samples")
    print(end - start)


@socketio.on('disconnect')
def ws_disconn():
    c = db.decr('connected')
    socketio.emit('msg', {'count': c})


if __name__ == '__main__':

    os.system("rhfCpp/main.out foo.exr -h bar.exr -d 0.8 static/rhf/image_filt.exr")
    # os.system("rhfCpp/exrtopng static/rhf/image_filt.exr  static/rhf/image_filt.png")

    
    print("The server has sterted");

    socketio.run(app, "0.0.0.0", port=5000)
