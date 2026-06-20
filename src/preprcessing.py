import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 160)

engine.say("Hello Gaurav. This is a speech test.")
engine.runAndWait()
