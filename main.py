"""
Bot de Discord + IA con Teachable Machine 🧠🤖
-----------------------------------------------
Carcasa con huecos para que completes.
"""

import discord
from discord.ext import commands
from keras.models import load_model
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

# importar lo necesario para IA (keras, numpy, PIL)
TOKEN = os.getenv("TOKEN_DISCORD")
PREFIX = "!"

# Habilitar intents
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=PREFIX, intents=intents)


print("TOKEN:", TOKEN)
# =========================
# Cargar modelo y etiquetas
# =========================
MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"

model = load_model(MODEL_PATH, compile=False)
class_names = open(LABELS_PATH, "r").readlines()
# 👉 aquí deben cargar keras_Model.h5 y labels.txt
#    pista: load_model(), open(), readlines()

# =========================
# Función de predicción
# =========================
def predict_image(image_path):
    # abrir imagen con PIL
    imagen = Image.open(image_path).convert("RGB")
    # redimensionar a 224x224
    imagen = imagen.resize((224, 224))
    # convertir a array numpy
    imagen_array = np.asarray(imagen)
    # normalizar valores (x/127.5 - 1)
    imagen_array = (imagen_array.astype(np.float32) / 127.5) - 1
    # preparar el input con shape (1, 224, 224, 3)
    imagen_array = np.expand_dims(imagen_array, axis=0)
    # usar model.predict()
    prediction = model.predict(imagen_array)
    # devolver clase y confianza
    index = np.argmax(prediction)
    confidence = prediction[0][index]
    label = class_names[index].strip()
    return label , confidence


# =========================
# Eventos y comandos
# =========================
@bot.event
async def on_ready():
    print(f"✅ Bot conectado como {bot.user}")

# =========================
# Eventos
# =========================
@bot.event
async def on_ready():
    print(f"✅ Bot conectado como {bot.user}")

# =========================
# Comando analizar
# =========================
@bot.command(name="analizar")
async def analizar(ctx):
    if not ctx.message.attachments:
        await ctx.send("❌ Debes subir una imagen.")
        return

    attachment = ctx.message.attachments[0]

    if not attachment.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        await ctx.send("❌ El archivo debe ser una imagen.")
        return

    image_path = "temp_image.jpg"
    await attachment.save(image_path)

    try:
        label, confidence = predict_image(image_path)

        await ctx.send(
            f"🧠 **Resultado del análisis**\n"
            f"📌 Esto es un **{label}**\n"
            f"🔍 Confianza: **{confidence:.2f}**"
        )

    except Exception as e:
        await ctx.send("⚠️ Error al analizar la imagen.")
        print(e)

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


# =========================
# Iniciar bot
# =========================
bot.run(TOKEN)
