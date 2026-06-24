from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from uuid import uuid4
import os
import cv2
import requests

# =========================================
# CONFIGURACIÓN GENERAL
# =========================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

PROCESSED_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "procesadas"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# Tamaño máximo permitido: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# =========================================
# CONFIGURACIÓN DE OLLAMA
# =========================================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/generate"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3"
)

# =========================================
# ÚLTIMO ANÁLISIS REALIZADO
# =========================================

ultimo_analisis = {
    "hay_analisis": False,
    "resultado": "Todavía no se ha analizado ninguna imagen.",
    "porcentaje": 0,
    "defectos_detectados": 0,
    "imagen": None,
    "procesada": None,
    "zonas_detectadas": [],
    "descripcion": "No hay análisis disponible todavía."
}

# =========================================
# CREAR CARPETAS
# =========================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    PROCESSED_FOLDER,
    exist_ok=True
)

# =========================================
# EXTENSIONES PERMITIDAS
# =========================================

EXTENSIONES_PERMITIDAS = {
    "png",
    "jpg",
    "jpeg",
    "webp"
}


def archivo_permitido(nombre_archivo):
    """
    Verifica que el archivo tenga una extensión válida.
    """

    return (
        "." in nombre_archivo
        and nombre_archivo.rsplit(".", 1)[1].lower()
        in EXTENSIONES_PERMITIDAS
    )


# =========================================
# PÁGINA PRINCIPAL
# =========================================

@app.route("/", methods=["GET", "POST"])
def index():

    global ultimo_analisis

    resultado = None
    imagen = None
    procesada = None
    porcentaje = 0
    defectos_detectados = 0

    if request.method == "POST":

        # =========================================
        # VALIDAR ARCHIVO
        # =========================================

        if "imagen" not in request.files:

            return render_template(
                "index.html",
                resultado="❌ No se seleccionó ninguna imagen",
                imagen=None,
                procesada=None,
                porcentaje=0,
                defectos_detectados=0
            )

        archivo = request.files["imagen"]

        if archivo.filename == "":

            return render_template(
                "index.html",
                resultado="❌ El archivo seleccionado no es válido",
                imagen=None,
                procesada=None,
                porcentaje=0,
                defectos_detectados=0
            )

        if not archivo_permitido(archivo.filename):

            return render_template(
                "index.html",
                resultado="❌ Formato no permitido. Usa PNG, JPG, JPEG o WEBP",
                imagen=None,
                procesada=None,
                porcentaje=0,
                defectos_detectados=0
            )

        try:

            # =========================================
            # GENERAR NOMBRE SEGURO
            # =========================================

            nombre_original = secure_filename(
                archivo.filename
            )

            extension = nombre_original.rsplit(
                ".",
                1
            )[1].lower()

            nombre = f"{uuid4().hex}.{extension}"

            ruta_original = os.path.join(
                app.config["UPLOAD_FOLDER"],
                nombre
            )

            archivo.save(
                ruta_original
            )

            # =========================================
            # LEER IMAGEN
            # =========================================

            img = cv2.imread(
                ruta_original
            )

            if img is None:

                return render_template(
                    "index.html",
                    resultado="❌ No se pudo leer la imagen",
                    imagen=None,
                    procesada=None,
                    porcentaje=0,
                    defectos_detectados=0
                )

            # =========================================
            # REDIMENSIONAR IMAGEN
            # =========================================

            img = cv2.resize(
                img,
                (800, 500)
            )

            # =========================================
            # CONVERTIR A ESCALA DE GRISES
            # =========================================

            gris = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            # =========================================
            # REDUCIR RUIDO
            # =========================================

            blur = cv2.GaussianBlur(
                gris,
                (5, 5),
                0
            )

            # =========================================
            # DETECTAR BORDES
            # =========================================

            bordes = cv2.Canny(
                blur,
                80,
                200
            )

            # =========================================
            # ENCONTRAR CONTORNOS
            # =========================================

            contornos, _ = cv2.findContours(
                bordes,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # =========================================
            # CALCULAR PORCENTAJE
            # =========================================

            pixeles_detectados = cv2.countNonZero(
                bordes
            )

            altura, ancho = gris.shape

            total_pixeles = altura * ancho

            if total_pixeles > 0:

                porcentaje = round(
                    (
                        pixeles_detectados
                        / total_pixeles
                    ) * 100,
                    2
                )

            # =========================================
            # MARCAR POSIBLES DEFECTOS
            # =========================================

            zonas_detectadas = []

            for contorno in contornos:

                area = cv2.contourArea(
                    contorno
                )

                # Ignorar contornos muy pequeños
                if area > 120:

                    defectos_detectados += 1

                    x, y, w, h = cv2.boundingRect(
                        contorno
                    )

                    zonas_detectadas.append({
                        "x": int(x),
                        "y": int(y),
                        "ancho": int(w),
                        "alto": int(h),
                        "area": round(float(area), 2)
                    })

                    cv2.rectangle(
                        img,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        3
                    )

                    posicion_texto = max(
                        y - 10,
                        25
                    )

                    cv2.putText(
                        img,
                        "DEFECTO",
                        (x, posicion_texto),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

            # =========================================
            # GENERAR RESULTADO
            # =========================================

            if porcentaje > 3:

                resultado = "✖ DEFECTO DETECTADO"

                descripcion = (
                    f"VisionIA detectó posibles defectos en la imagen. "
                    f"El nivel de daño fue de {porcentaje}% y se encontraron "
                    f"aproximadamente {defectos_detectados} zonas marcadas."
                )

            else:

                resultado = "✔ PRODUCTO CORRECTO"

                descripcion = (
                    f"VisionIA no detectó un nivel alto de daño. "
                    f"El porcentaje encontrado fue de {porcentaje}% y se marcaron "
                    f"{defectos_detectados} posibles zonas."
                )

            # =========================================
            # GUARDAR IMAGEN PROCESADA
            # =========================================

            ruta_procesada = os.path.join(
                app.config["PROCESSED_FOLDER"],
                nombre
            )

            guardado = cv2.imwrite(
                ruta_procesada,
                img
            )

            if not guardado:

                raise RuntimeError(
                    "No se pudo guardar la imagen procesada"
                )

            imagen = nombre
            procesada = nombre

            # =========================================
            # GUARDAR ÚLTIMO ANÁLISIS PARA OLLAMA
            # =========================================

            ultimo_analisis = {
                "hay_analisis": True,
                "resultado": resultado,
                "porcentaje": porcentaje,
                "defectos_detectados": defectos_detectados,
                "imagen": imagen,
                "procesada": procesada,
                "zonas_detectadas": zonas_detectadas[:10],
                "descripcion": descripcion
            }

            print("\n✅ Imagen procesada correctamente")
            print(f"📊 Porcentaje detectado: {porcentaje}%")
            print(
                f"🔍 Posibles defectos encontrados: "
                f"{defectos_detectados}"
            )

        except Exception as error:

            print("\n❌ ERROR AL PROCESAR LA IMAGEN:")
            print(error)

            resultado = (
                "❌ Ocurrió un error al procesar la imagen"
            )

    return render_template(
        "index.html",
        resultado=resultado,
        imagen=imagen,
        procesada=procesada,
        porcentaje=porcentaje,
        defectos_detectados=defectos_detectados
    )


# =========================================
# CONSULTAR ÚLTIMO ANÁLISIS
# =========================================

@app.route("/ultimo-analisis", methods=["GET"])
def obtener_ultimo_analisis():

    return jsonify(
        ultimo_analisis
    )


# =========================================
# CHAT CON OLLAMA
# =========================================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        # Permite recibir datos desde formulario o JSON
        datos_json = request.get_json(
            silent=True
        ) or {}

        mensaje = (
            request.form.get("mensaje")
            or datos_json.get("mensaje")
            or ""
        ).strip()

        if not mensaje:

            return jsonify({
                "respuesta": "❌ Escribe un mensaje antes de enviarlo"
            }), 400

        print("\n📩 MENSAJE DEL USUARIO:")
        print(mensaje)

        # =========================================
        # CONTEXTO DEL ÚLTIMO ANÁLISIS
        # =========================================

        zonas_texto = "No hay zonas registradas."

        if ultimo_analisis["zonas_detectadas"]:

            zonas_texto = ""

            for i, zona in enumerate(
                ultimo_analisis["zonas_detectadas"],
                start=1
            ):

                zonas_texto += (
                    f"Zona {i}: "
                    f"x={zona['x']}, "
                    f"y={zona['y']}, "
                    f"ancho={zona['ancho']}, "
                    f"alto={zona['alto']}, "
                    f"área={zona['area']}.\n"
                )

        contexto_analisis = f"""
Último análisis realizado por VisionIA:

¿Ya se analizó una imagen?: {ultimo_analisis["hay_analisis"]}
Resultado: {ultimo_analisis["resultado"]}
Porcentaje de daño detectado: {ultimo_analisis["porcentaje"]}%
Cantidad aproximada de defectos encontrados: {ultimo_analisis["defectos_detectados"]}
Imagen original guardada: {ultimo_analisis["imagen"]}
Imagen procesada guardada: {ultimo_analisis["procesada"]}

Descripción del análisis:
{ultimo_analisis["descripcion"]}

Zonas detectadas:
{zonas_texto}

Nota técnica:
El análisis fue realizado con OpenCV. El proceso usado fue:
1. Redimensionar la imagen.
2. Convertir la imagen a escala de grises.
3. Aplicar reducción de ruido con GaussianBlur.
4. Detectar bordes con Canny.
5. Buscar contornos.
6. Dibujar rectángulos rojos sobre las posibles zonas defectuosas.
"""

        instrucciones = f"""
Eres VisionIA, un asistente virtual del proyecto VisionIA.

Información del proyecto:
VisionIA es una aplicación web hecha con Flask, OpenCV y Ollama.
Su función principal es permitir que el usuario suba una imagen de un producto,
analizarla y marcar posibles defectos con rectángulos rojos.

Tu especialidad:
- Inteligencia artificial
- Visión computacional
- OpenCV
- Manufactura
- Calidad industrial
- Automatización industrial
- Detección de defectos en productos

Información disponible del último análisis:
{contexto_analisis}

Reglas para responder:
- Responde siempre en español.
- Usa un lenguaje claro, corto y fácil de entender.
- Si el usuario pregunta sobre la imagen, responde usando el último análisis realizado.
- Si todavía no se ha analizado ninguna imagen, dile que primero debe subir y analizar una imagen.
- No digas que estás viendo directamente la imagen.
- Explica que respondes con base en el análisis realizado por VisionIA con OpenCV.
- No inventes datos que no estén en el análisis.
- Si te preguntan "qué analizaste", menciona el resultado, porcentaje, cantidad de defectos y proceso usado.
"""

        datos_ollama = {
            "model": OLLAMA_MODEL,
            "system": instrucciones,
            "prompt": mensaje,
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        }

        respuesta_ollama = requests.post(
            OLLAMA_URL,
            json=datos_ollama,
            timeout=(5, 120)
        )

        # El modelo no está instalado
        if respuesta_ollama.status_code == 404:

            return jsonify({
                "respuesta": (
                    f"❌ El modelo '{OLLAMA_MODEL}' no está instalado. "
                    f"Ejecuta en la terminal: "
                    f"ollama pull {OLLAMA_MODEL}"
                )
            }), 503

        respuesta_ollama.raise_for_status()

        datos_respuesta = respuesta_ollama.json()

        respuesta = datos_respuesta.get(
            "response",
            ""
        ).strip()

        if not respuesta:

            return jsonify({
                "respuesta": (
                    "❌ Ollama respondió, pero no generó texto"
                )
            }), 502

        print("\n✅ RESPUESTA DE OLLAMA:")
        print(respuesta)

        return jsonify({
            "respuesta": respuesta
        })

    except requests.exceptions.ConnectionError:

        print("\n❌ No se pudo conectar con Ollama")

        return jsonify({
            "respuesta": (
                "❌ No se pudo conectar con Ollama. "
                "Asegúrate de que Ollama esté instalado y abierto."
            )
        }), 503

    except requests.exceptions.Timeout:

        print("\n❌ Ollama tardó demasiado en responder")

        return jsonify({
            "respuesta": (
                "❌ Ollama tardó demasiado en responder. "
                "Intenta nuevamente."
            )
        }), 504

    except requests.exceptions.RequestException as error:

        print("\n❌ ERROR DE CONEXIÓN CON OLLAMA:")
        print(error)

        return jsonify({
            "respuesta": (
                "❌ Ocurrió un problema al comunicarse con Ollama"
            )
        }), 502

    except ValueError:

        print("\n❌ Ollama devolvió una respuesta inválida")

        return jsonify({
            "respuesta": (
                "❌ Ollama devolvió una respuesta que no se pudo leer"
            )
        }), 502

    except Exception as error:

        print("\n❌ ERROR EN EL CHAT:")
        print(error)

        return jsonify({
            "respuesta": (
                f"❌ Ocurrió un error en el chat: {str(error)}"
            )
        }), 500


# =========================================
# ARCHIVO DEMASIADO GRANDE
# =========================================

@app.errorhandler(413)
def archivo_demasiado_grande(error):

    return render_template(
        "index.html",
        resultado="❌ La imagen supera el límite de 10 MB",
        imagen=None,
        procesada=None,
        porcentaje=0,
        defectos_detectados=0
    ), 413


# =========================================
# EJECUTAR APLICACIÓN
# =========================================

if __name__ == "__main__":

    print("\n🚀 VisionIA iniciado")
    print(f"🤖 Modelo de Ollama: {OLLAMA_MODEL}")
    print("🌐 Dirección: http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
