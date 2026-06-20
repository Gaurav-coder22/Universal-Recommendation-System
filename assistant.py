import speech_recognition as sr
import subprocess
import winsound
import os
import time
import webbrowser
import pyautogui
import pywhatkit
import json
from difflib import get_close_matches

from slm_chat import slm_query
from universal_recommender import (
    recommend,
    recommend_music_itunes,
    METADATA
)

# --------------------------------------------------
# FAST PIPER TTS
# --------------------------------------------------

PIPER_EXE = r"D:\machine learning\piper\piper.exe"
MODEL = r"D:\machine learning\voices\en_US-hfc_female-medium.onnx"
TEMP_WAV = r"D:\machine learning\voices\tts_output.wav"

def say(text):
    print("Assistant:", text)

    process = subprocess.Popen(
        [
            PIPER_EXE,
            "--model", MODEL,
            "--output_file", TEMP_WAV,
            "--length_scale", "0.80",
            "--noise_scale", "0.25",
            "--noise_w_scale", "0.25",
            "--sample_rate", "16000",
            "--speed", "1.55"
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    process.communicate(text)
    winsound.PlaySound(TEMP_WAV, winsound.SND_FILENAME)


# --------------------------------------------------
# STABLE LISTENING
# --------------------------------------------------

def listen():
    r = sr.Recognizer()
    r.dynamic_energy_threshold = False
    r.energy_threshold = 260
    r.pause_threshold = 0.6

    with sr.Microphone() as source:
        print("\n🎤 Listening...")
        try:
            audio = r.listen(source, timeout=None, phrase_time_limit=5)
        except:
            return ""

    try:
        text = r.recognize_google(audio, language='en-IN')
        print("You:", text)
        return text.lower()
    except:
        return ""


# --------------------------------------------------
# RULE-BASED UNDERSTANDING ENGINE
# --------------------------------------------------

def slm_understand(text):
    t = text.lower().strip()

    # WHATSAPP
    if "send message" in t or "send whatsapp" in t or "whatsapp message" in t:
        words = t.split()
        contact, message = "", ""

        if "to" in words:
            idx = words.index("to")
            if idx + 1 < len(words):
                contact = words[idx + 1]
                if idx + 2 < len(words):
                    message = " ".join(words[idx + 2:])

        return {
            "domain": "whatsapp",
            "query": "",
            "contact": contact,
            "message": message
        }

    # MOVIES
    if "movie" in t or "film" in t:
        q = (
            t.replace("recommend", "")
             .replace("movie", "")
             .replace("movies", "")
             .replace("film", "")
             .replace("like", "")
             .strip()
        )
        return {"domain": "movies", "query": q, "contact": "", "message": ""}

    # MUSIC
    if "music" in t or "song" in t:
        q = (
            t.replace("recommend", "")
             .replace("song", "")
             .replace("music", "")
             .replace("like", "")
             .strip()
        )
        return {"domain": "music", "query": q, "contact": "", "message": ""}

    # FOOD
    if "food" in t or "recipe" in t:
        return {"domain": "food", "query": t, "contact": "", "message": ""}

    # CLOTHES
    if any(w in t for w in ["clothes", "dress", "shirt"]):
        return {"domain": "clothes", "query": t, "contact": "", "message": ""}

    # PRODUCTS
    if "product" in t or "item" in t:
        return {"domain": "products", "query": t, "contact": "", "message": ""}

    # VIDEOS
    if "video" in t or "youtube" in t:
        return {"domain": "videos", "query": t, "contact": "", "message": ""}

    # GENERAL QUESTIONS → SLM
    if any(w in t for w in ["what", "why", "how", "explain", "summarize", "who", "meaning"]):
        return {"domain": "slm", "query": t, "contact": "", "message": ""}

    return {"domain": "unknown", "query": t, "contact": "", "message": ""}


# --------------------------------------------------
# WHATSAPP SENDER (RESEND FIX)
# --------------------------------------------------

PHONEBOOK = {
    "d": "+919198697798"
}

def send_whatsapp(contact, message):
    contact = contact.lower()

    if contact not in PHONEBOOK:
        say(f"I don't have a number for {contact}.")
        return

    number = PHONEBOOK[contact]
    say(f"Sending WhatsApp message to {contact}")

    url = f"https://web.whatsapp.com/send?phone={number}&text="
    webbrowser.open(url)

    say("Opening WhatsApp...")
    time.sleep(12)

    pyautogui.click(x=800, y=950)  # Click message box
    pyautogui.typewrite(message, interval=0.02)
    time.sleep(0.4)
    pyautogui.press("enter")

    say("Message sent successfully!")


# --------------------------------------------------
# MOVIE TITLE CLEANER
# --------------------------------------------------

def clean_movie_query(query):
    try:
        df = METADATA.get("movies")
        if df is None:
            return query

        titles = df["title"].astype(str).tolist()
        match = get_close_matches(query, titles, n=1, cutoff=0.3)
        return match[0] if match else query

    except:
        return query


# --------------------------------------------------
# MAIN ASSISTANT LOOP
# --------------------------------------------------

def assistant():
    say("Hello! How can I help you today?")

    while True:
        user = listen()
        if not user:
            continue

        if user in ["exit", "stop", "quit", "bye"]:
            say("Goodbye!")
            break

        parsed = slm_understand(user)
        print("AI Parsed:", parsed)

        domain = parsed["domain"]
        query = parsed["query"]
        contact = parsed.get("contact", "")
        message = parsed.get("message", "")

        # --------------------------------------------------
        # ASK AGAIN IF USER GIVES NO QUERY
        # --------------------------------------------------
        if domain in ["movies", "music", "food", "clothes", "products", "videos"]:
            if query.strip() == "":
                say(f"Which {domain[:-1]} should I search for?")
                query = listen().strip()

                if not query:
                    say("I did not hear anything.")
                    continue

        # --------------------------------------------------
        # WHATSAPP MODE
        # --------------------------------------------------
        if domain == "whatsapp":
            if not contact:
                say("Whom should I message?")
                contact = listen().strip()

            if not message:
                say("What is the message?")
                message = listen().strip()

            send_whatsapp(contact, message)
            continue

        # --------------------------------------------------
        # GENERAL QUESTIONS → SLM
        # --------------------------------------------------
        if domain == "slm":
            say("Processing your question...")
            response = slm_query(query)
            say(response)
            continue

        # --------------------------------------------------
        # RECOMMENDATION SYSTEM
        # --------------------------------------------------
        if domain == "movies":
            query = clean_movie_query(query)
            results = recommend("movies", query, 5)

        elif domain == "music":
            results = recommend_music_itunes(query, 5)

        elif domain in ["food", "clothes", "products", "videos"]:
            results = recommend(domain, query, 5)

        else:
            say("I am not sure what you meant.")
            continue

        if not results or isinstance(results, str):
            say("I couldn't find anything.")
            continue

        say("Here are your recommendations:")
        for i, item in enumerate(results, 1):
            say(f"{i}. {item.get('title', 'Unknown')}")


assistant()
