import io
from fastapi import Response
import numpy as np
from pydantic import BaseModel
import torch
import os
import scipy
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)

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
        audio_wav_bytes = audio_bytes

        # Конвертируйте WAV в MP3
        audio_wav = AudioSegment.from_wav(io.BytesIO(audio_wav_bytes))
        audio_mp3_bytes = io.BytesIO()
        audio_wav.export(audio_mp3_bytes, format="mp3", bitrate="192k")

        # Теперь audio_mp3_bytes содержит аудио в формате MP3
        audio_mp3_bytes.seek(0)
        audio_mp3_content = audio_mp3_bytes.read()

        # Возвращаем ответ с MP3
        if len(audio_mp3_content) > 0:
            logger.info("Audio was generated")
        return Response(content=audio_mp3_content)
    except:
        logger.error("Couldn't generate audio")
        return {"message":"ОШИБКА: генерация аудио не удалась"}

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