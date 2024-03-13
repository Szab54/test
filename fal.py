import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import time
from openai import OpenAI
import pygame
import threading

import zene  # Saját modul

client = OpenAI(api_key='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')
import pyttsx3
import speech_recognition as sr

# Hangparaméterek
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

activate = False
speech = ""

# PyAudio inicializálása
audio = pyaudio.PyAudio()

# Stream létrehozása
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

# Hanghullám megjelenítése
plt.ion()  # Interaktív üzemmód bekapcsolása
fig, ax = plt.subplots()
x = np.arange(0, 2 * CHUNK, 2)
line, = ax.plot(x, np.random.rand(CHUNK))

ax.set_ylim(-32768, 32768)
ax.set_xlim(0, CHUNK)

def plot_audio_stream():
    while True:
        data = stream.read(CHUNK)
        data_int = np.frombuffer(data, dtype=np.int16)
        line.set_ydata(data_int)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)  # Várakozás a következő frissítésig

# Szál a hanghullámok megjelenítéséhez
plot_thread = threading.Thread(target=plot_audio_stream)
plot_thread.daemon = True
plot_thread.start()

def chat_with_gpt(prompt):
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=0.7,
        max_tokens=500
    )
    return response.choices[0].text.strip()

def text_to_speech_with_openai_tts(text, adat):
    response = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=adat,
    )
    response.stream_to_file("chatgpt.mp3")

def recognize_speech():
    global activate, speech

    recognizer = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            print("Kérlek, beszélj!")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

            try:
                speech = recognizer.recognize_google(audio, language='hu-HU')
                print("Felismert beszéd: " + speech)

                if speech == "arló segíts nekem":
                    print(" --------------------> AKTIVÁLVA <--------------------")
                    activate = True
                elif "viszlát" in speech:
                    print(" --------------------> DEAKTIVÁLVA <--------------------")
                    adat = "Viszlát!"
                    text_to_speech_with_openai_tts(response, adat)
                    pygame.init()
                    pygame.mixer.music.load('chatgpt.mp3')
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)

                    activate = False

                elif speech == "kilépés":
                    print(" --------------------> EXIT  <--------------------")
                    exit()

                elif "zene" in speech:
                    time.sleep(1)
                    zene.zene1()

                elif "zene stop" in speech:
                    zene.stop()
                    time.sleep(1)
                    response = "Rendben a zenét meg állítottam!"
                    print(response)
                    adat = response
                    text_to_speech_with_openai_tts(response, adat)

                if activate:
                    prompt = f"Felhasználó: {speech}\nChatGPT:"
                    response = chat_with_gpt(prompt)
                    print("Felhasználó:", speech)
                    print("Árló:", response)
                    adat = response
                    text_to_speech_with_openai_tts(response, adat)

                    pygame.init()
                    pygame.mixer.music.load('chatgpt.mp3')
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                else:
                    print("A bot deaktiválva van!")

            except sr.UnknownValueError:
                print("Nem sikerült felismerni a beszédet.")
            except sr.RequestError as e:
                print("Hiba a Google API hívás közben; {0}".format(e))

if __name__ == "__main__":
    recognize_speech()

    # Stream és PyAudio lezárása
    stream.stop_stream()
    stream.close()
    audio.terminate()
