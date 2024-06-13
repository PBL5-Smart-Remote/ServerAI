import os
from flask import Flask, Response, send_file
import json
from util.AI.main import prediction

from pydub import AudioSegment
import requests
from colorama import Fore, Back, Style

import os
from datetime import datetime
import pytz


utc_now = datetime.now(pytz.utc)
utc_plus_7 = pytz.timezone('Asia/Bangkok')
local_time = utc_now.astimezone(utc_plus_7)
formatted_time = local_time.strftime('%d_%m_%Y-%H-%M-%S')
file_name = f"output_{formatted_time}"


class SiteController:
    def getLabel(request):
        # print(request.)
        audio_file = request.files['audio']
        if audio_file.filename == '':
            data = {"text": "NOT FOUND FILES AUDIO/WAV"}
            return Response(response=json.dumps(data), status=304,
                            mimetype='application/json')

        print(Fore.YELLOW, 'Audio name: ', f'{file_name}.wav', Fore.WHITE)
        audio_file.save(
            os.getcwd() + f'\\src\\resources\\audio\\{file_name}.wav')

        label, decodedLabel = prediction(file_name)
        print(Fore.RED, '> Label Predicted: ' + label)
        print(Fore.BLUE, '> Calling API to active this action: ' + label)
        # Send request to ServerFirmware
        # body Json
        # Prepare the JSON data
        json_data = {"label": decodedLabel}
        url = "https://api.tugino.com:4501/devices/updateByAudio"
        response = requests.put(url, json=json_data)

        print(Fore.GREEN, '> Call API successfully', Fore.WHITE)

        if response.status_code == 200:
            resp = Response(response=json.dumps({
                'label': label
            }), status=200, mimetype='application/json')
        else:
            resp = Response(response=json.dumps('failed'), status=400,
                            mimetype='application/json')

        return resp
