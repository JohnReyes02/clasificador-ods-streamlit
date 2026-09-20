"""Aplicación web para clasificar textos según los ODS."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


DIRECTORIO_APLICACION = Path(__file__).resolve().parent
DIRECTORIO_REPOSITORIO = DIRECTORIO_APLICACION.parents[1]
RUTA_MODELO = DIRECTORIO_APLICACION / "mejor_modelo_clasificacion.joblib"
DIRECTORIO_ICONOS = DIRECTORIO_REPOSITORIO / "assets" / "ods"

ODS = {
    1: ("Fin de la pobreza", "ods-01-fin-de-la-pobreza.png"),
    2: ("Hambre cero", "ods-02-hambre-cero.png"),
    3: ("Salud y bienestar", "ods-03-salud-y-bienestar.png"),
    4: ("Educación de calidad", "ods-04-educacion-de-calidad.png"),
    5: ("Igualdad de género", "ods-05-igualdad-de-genero.png"),
    6: ("Agua limpia y saneamiento", "ods-06-agua-limpia-y-saneamiento.png"),
    7: ("Energía asequible y no contaminante", "ods-07-energia-asequible-y-no-contaminante.png"),
    8: ("Trabajo decente y crecimiento económico", "ods-08-trabajo-decente-y-crecimiento-economico.png"),
    9: ("Industria, innovación e infraestructura", "ods-09-industria-innovacion-e-infraestructura.png"),
    10: ("Reducción de las desigualdades", "ods-10-reduccion-de-las-desigualdades.png"),
    11: ("Ciudades y comunidades sostenibles", "ods-11-ciudades-y-comunidades-sostenibles.png"),
    12: ("Producción y consumo responsables", "ods-12-produccion-y-consumo-responsables.png"),
    13: ("Acción por el clima", "ods-13-accion-por-el-clima.png"),
    14: ("Vida submarina", "ods-14-vida-submarina.png"),
    15: ("Vida de ecosistemas terrestres", "ods-15-vida-de-ecosistemas-terrestres.png"),
    16: ("Paz, justicia e instituciones sólidas", "ods-16-paz-justicia-e-instituciones-solidas.png"),
    17: ("Alianzas para lograr los objetivos", "ods-17-alianzas-para-lograr-los-objetivos.png"),
}

EJEMPLOS = {
    "Agua y saneamiento": (
        "Las comunidades rurales necesitan acceso continuo a agua potable, "
        "alcantarillado y servicios de saneamiento seguros."
    ),
    "Educación": (
        "Se propone garantizar educación inclusiva y de calidad, con escuelas "
        "seguras y oportunidades de aprendizaje para todos los estudiantes."
    ),
    "Acción climática": (
        "El municipio reducirá sus emisiones de gases de efecto invernadero y "
        "fortalecerá la adaptación frente a inundaciones y sequías."
    ),
    "Igualdad de género": (
        "La política busca eliminar la violencia contra las mujeres y asegurar "
        "igualdad de oportunidades y participación en cargos de liderazgo."
    ),
}


st.set_page_config(
    page_title="Clasificador de textos por ODS",
    page_icon="🌎",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1050px; padding-top: 2.4rem; padding-bottom: 3rem;}
        [data-testid="stForm"] {border: 1px solid #dce6e2; border-radius: 16px; padding: 1.2rem;}
        .encabezado-menor {color: #087f5b; font-size: .78rem; font-weight: 700; letter-spacing: .08em;
                  margin-bottom: .35rem; text-transform: uppercase;}
        .subtitulo {color: #52616b; font-size: 1.08rem; max-width: 760px; margin-bottom: 1.4rem;}
        .tarjeta-resultado {background: linear-gradient(135deg, #f2fbf7 0%, #eef6fb 100%);
                      border: 1px solid #cfe4da; border-radius: 18px; padding: 1.3rem 1.5rem;}
        .etiqueta-resultado {color: #52616b; font-size: .8rem; font-weight: 700;
                       letter-spacing: .06em; text-transform: uppercase;}
        .titulo-resultado {color: #153b2f; font-size: 1.55rem; font-weight: 750; margin: .2rem 0;}
        .confianza-resultado {color: #087f5b; font-size: 1rem; font-weight: 650;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando el modelo de clasificación…")
def cargar_modelo():
    """Carga y conserva en memoria el modelo completo de clasificación.

    Streamlit almacena el resultado en caché para evitar leer el archivo y
    reconstruir el flujo TF-IDF → SVD → regresión logística en cada
    interacción del usuario.

    Devuelve:
        El flujo de clasificación previamente entrenado.

    Excepciones:
        FileNotFoundError: Si el archivo del modelo no existe en la ruta esperada.
    """
    if not RUTA_MODELO.is_file():
        raise FileNotFoundError(f"No se encontró el modelo en {RUTA_MODELO}")
    return joblib.load(RUTA_MODELO)


def predecir(texto: str) -> tuple[int, float, pd.DataFrame]:
    """Estima el ODS que guarda mayor relación con un texto.

    Parámetros:
        texto: Fragmento en español que se desea clasificar.

    Devuelve:
        Una tupla con el número del ODS predicho, la confianza de la predicción
        y una tabla con las cinco categorías que obtuvieron mayor probabilidad.
    """
    modelo = cargar_modelo()
    probabilidades = modelo.predict_proba([texto])[0]
    clases = [int(valor) for valor in modelo.classes_]
    puntuaciones = sorted(
        zip(clases, probabilidades),
        key=lambda elemento: elemento[1],
        reverse=True,
    )
    prediccion, confianza = puntuaciones[0]
    clasificacion = pd.DataFrame(
        {
            "ODS": [
                f"ODS {numero} · {ODS[numero][0]}"
                for numero, _ in puntuaciones[:5]
            ],
            "Probabilidad": [
                probabilidad for _, probabilidad in puntuaciones[:5]
            ],
        }
    ).set_index("ODS")
    return prediccion, float(confianza), clasificacion


if "texto_entrada" not in st.session_state:
    st.session_state.texto_entrada = ""

st.markdown(
    '<div class="encabezado-menor">Aprendizaje automático supervisado · Microproyecto 2</div>',
    unsafe_allow_html=True,
)
st.title("Clasificador de textos por Objetivo de Desarrollo Sostenible")
st.caption(
    "Autores: Edwin John Fredy Reyes Aguirre y Jhon Jairo Hernández Quintero "
    "· Universidad de los Andes"
)
st.markdown(
    '<div class="subtitulo">Escribe una iniciativa, necesidad o fragmento de documento. '
    "El modelo analizará su contenido y estimará el ODS con mayor relación semántica.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Ejemplos")
    ejemplo_seleccionado = st.selectbox(
        "Selecciona un tema",
        ["— Elegir —", *EJEMPLOS],
        label_visibility="collapsed",
    )
    if st.button(
        "Usar este ejemplo",
        use_container_width=True,
        disabled=ejemplo_seleccionado == "— Elegir —",
    ):
        st.session_state.texto_entrada = EJEMPLOS[ejemplo_seleccionado]
        st.rerun()

    st.divider()
    st.markdown("**Acerca del modelo**")
    st.caption(
        "Flujo completo de procesamiento con TF-IDF, reducción dimensional "
        "mediante SVD y "
        "regresión logística. Fue entrenado con las 16 clases presentes en el "
        "conjunto de datos (ODS 1 a 16)."
    )

with st.form("formulario_clasificacion"):
    texto = st.text_area(
        "Texto para analizar",
        key="texto_entrada",
        height=180,
        max_chars=5000,
        placeholder="Ejemplo: El programa instalará sistemas de energía solar en comunidades rurales…",
        help="Puedes ingresar hasta 5.000 caracteres.",
    )
    enviado = st.form_submit_button(
        "Analizar texto",
        type="primary",
        use_container_width=True,
    )

if enviado:
    texto_limpio = " ".join(texto.split())
    if len(texto_limpio) < 20:
        st.warning("Escribe un texto un poco más descriptivo (al menos 20 caracteres).")
    else:
        try:
            numero_ods, confianza, clasificacion = predecir(texto_limpio)
        except Exception as excepcion:
            st.error("No fue posible cargar el modelo o generar la predicción.")
            with st.expander("Detalle técnico"):
                st.code(f"{type(excepcion).__name__}: {excepcion}")
        else:
            st.divider()
            columna_imagen, columna_resultado = st.columns(
                [1, 3],
                vertical_alignment="center",
            )
            with columna_imagen:
                ruta_icono = DIRECTORIO_ICONOS / ODS[numero_ods][1]
                if ruta_icono.is_file():
                    st.image(str(ruta_icono), width=180)
            with columna_resultado:
                st.markdown(
                    f"""
                    <div class="tarjeta-resultado">
                        <div class="etiqueta-resultado">Predicción principal</div>
                        <div class="titulo-resultado">ODS {numero_ods} · {ODS[numero_ods][0]}</div>
                        <div class="confianza-resultado">Confianza estimada: {confianza:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.subheader("Otras categorías consideradas")
            st.bar_chart(
                clasificacion,
                horizontal=True,
                color="#0b846e",
                x_label="Probabilidad estimada",
                y_label="",
            )
            st.caption(
                "Las probabilidades reflejan la seguridad relativa del modelo, no una "
                "evaluación oficial de Naciones Unidas. Un texto puede relacionarse con varios ODS."
            )

with st.expander("Alcance y limitaciones"):
    st.markdown(
        """
        - El modelo fue entrenado con textos en español y asigna una sola categoría por entrada.
        - El ODS 17 no estaba presente en los datos de entrenamiento, por lo que no puede predecirse.
        - La clasificación sirve como apoyo exploratorio y debe complementarse con revisión humana.
        """
    )
