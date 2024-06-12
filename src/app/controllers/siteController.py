import os
from flask import Flask, Response, send_file
import json
from util.AI.main import prediction

from pydub import AudioSegment
import requests


class SiteController:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def getLabel(request):
        # print(request.)
        audio_file = request.files['audio']
        print(audio_file)
        if audio_file.filename == '':
            data = {"text": "NOT FOUND FILES AUDIO/WAV"}
            return Response(response=json.dumps(data), status=304,
                            mimetype='application/json')
        print(os.getcwd())
        audio_file.save(
            os.getcwd() + ('\\src\\resources\\audio\\output.wav'))

        label = prediction()
        print(label)

        # Send request to ServerFirmware
        # body Json
        # Prepare the JSON data
        json_data = {"label": label, "idRoom": request.form['idRoom']}
        url = "https://api.tugino.com:4501/devices/updateByAudio"

        response = requests.put(url, json=json_data)

        if response.status_code == 200:
            resp = Response(response=json.dumps('successful'), status=200,
                            mimetype='application/json')
        else:
            resp = Response(response=json.dumps('failed'), status=400,
                            mimetype='application/json')
        # return Response(response=json.dumps('successful'), status=200,
        #                 mimetype='application/json')

        return resp
