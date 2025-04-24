from flask import Flask, render_template, Response, jsonify, request
from sign_to_text_flask import get_frame, add_space, reset_sentence
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        frame, _ = get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_word')
def get_word():
    _, word = get_frame()
    return jsonify({'word': word})

@app.route('/add_space', methods=['POST'])
def space():
    add_space()
    return '', 204

@app.route('/reset', methods=['POST'])
def reset():
    reset_sentence()
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)