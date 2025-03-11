import asyncio
import os
import time
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

import tts


app = FastAPI()
semaphore = asyncio.Semaphore(30)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Text(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"Это сервер для ТТС."}


@app.post("/audio")
async def generate_audio(input_text: Text):

    #Если запрос находится в очереди больше 400 секунд - возвращает ошибку:
    start_time = time.time()

    async with semaphore:
        wait_time = time.time() - start_time
        if wait_time > 400:
            start_time = time.time()
            return {"message": "ОШИБКА: Время ожидания запроса в очереди превысило 400 секунд."}
        
        #Если запрос в течении 240 секунд не обработан, возвращает ошибку:
        try:
            start = time.time()
            response = await asyncio.wait_for(asyncio.to_thread(tts.GenerateAudio, input_text), timeout=240)
            
            end = time.time()
            print('It took {} seconds to finish execution.'.format(round(end-start)))
            return response
        except asyncio.TimeoutError:
            return {"message":"ОШИБКА: Время ожидания запроса превысило 240 секунд."}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8002))  # Default to 8002 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)
