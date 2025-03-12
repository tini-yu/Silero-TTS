import io
import uuid
from fastapi import Response
import numpy as np
from pydantic import BaseModel
import torch
import os
import scipy

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt',
                                   local_file)  

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)


sample_rate = 24000
speaker = 'xenia'
put_accent=True
put_yo=True

class Message(BaseModel):
    message: str
    
def GenerateAudio(input_text: Message):
    input_json = input_text.model_dump()
    input_text = input_json['message']

    try:
        audio = model.apply_tts(text=input_text,
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo)
        
        audio_buffer = io.BytesIO()
        # Convert the tensor to a NumPy array
        if not isinstance(audio, np.ndarray):
            audio = audio.cpu().numpy()  # Use `.cpu()` if the tensor is on GPU

        # Normalize the audio if it's in float format (e.g., between -1 and 1)
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            audio = np.clip(audio, -1.0, 1.0)  # Ensure values are in range [-1, 1]
            audio = (audio * 32767).astype(np.int16)  # Scale to int16 for WAV format
        scipy.io.wavfile.write(audio_buffer, rate=sample_rate, data=audio)
        # audio.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_bytes = audio_buffer.read()
    except:
        return {"message":"ОШИБКА: генерация аудио не удалась"}

    # # Тело = текст + разделить + аудио
    # separator = "---AUDIO---"
    # body = input_text + separator
    # body_bytes = body.encode('utf-8') + audio_bytes

    return Response(content = audio_bytes)

    # output_directory = "C:/Users/Joe/Desktop/audio_guide/silero/audiofiles"
    # os.makedirs(output_directory, exist_ok=True)
    # unique_filename = f"tts_audio_{uuid.uuid4()}.wav"
    # file_path = os.path.join(output_directory, unique_filename)

    # # Convert the tensor to a NumPy array
    # if not isinstance(audio, np.ndarray):
    #     audio = audio.cpu().numpy()  # Use `.cpu()` if the tensor is on GPU

    # # Normalize the audio if it's in float format (e.g., between -1 and 1)
    # if audio.dtype == np.float32 or audio.dtype == np.float64:
    #     audio = np.clip(audio, -1.0, 1.0)  # Ensure values are in range [-1, 1]
    #     audio = (audio * 32767).astype(np.int16)  # Scale to int16 for WAV format
        
    # scipy.io.wavfile.write(file_path, rate=sample_rate,
    #                     data=audio)
    
    # return file_path