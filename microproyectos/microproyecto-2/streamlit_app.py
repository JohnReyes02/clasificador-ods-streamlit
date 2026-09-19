"""Aplicación web para clasificar textos según los ODS."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parents[1]
MODEL_PATH = APP_DIR / "mejor_modelo_clasificacion.joblib"
ICONS_DIR = REPO_DIR / "assets" / "ods"

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

EXAMPLES = {
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
        .eyebrow {color: #087f5b; font-size: .78rem; font-weight: 700; letter-spacing: .08em;
                  margin-bottom: .35rem; text-transform: uppercase;}
        .subtitle {color: #52616b; font-size: 1.08rem; max-width: 760px; margin-bottom: 1.4rem;}
        .result-card {background: linear-gradient(135deg, #f2fbf7 0%, #eef6fb 100%);
                      border: 1px solid #cfe4da; border-radius: 18px; padding: 1.3rem 1.5rem;}
        .result-label {color: #52616b; font-size: .8rem; font-weight: 700;
                       letter-spacing: .06em; text-transform: uppercase;}
        .result-title {color: #153b2f; font-size: 1.55rem; font-weight: 750; margin: .2rem 0;}
        .result-confidence {color: #087f5b; font-size: 1rem; font-weight: 650;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Cargando el modelo de clasificación…")
def load_model():
    """Load the complete TF-IDF → SVD → logistic regression pipeline once."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict(text: str) -> tuple[int, float, pd.DataFrame]:
    """Return the predicted ODS, its probability, and the five best options."""
    model = load_model()
    probabilities = model.predict_proba([text])[0]
    classes = [int(value) for value in model.classes_]
    scores = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    prediction, confidence = scores[0]
    ranking = pd.DataFrame(
        {
            "ODS": [f"ODS {number} · {ODS[number][0]}" for number, _ in scores[:5]],
            "Probabilidad": [probability for _, probability in scores[:5]],
        }
    ).set_index("ODS")
    return prediction, float(confidence), ranking


if "input_text" not in st.session_state:
    st.session_state.input_text = ""

st.markdown('<div class="eyebrow">Machine learning no supervisado · Microproyecto 2</div>', unsafe_allow_html=True)
st.title("Clasificador de textos por Objetivo de Desarrollo Sostenible")
st.markdown(
    '<div class="subtitle">Escribe una iniciativa, necesidad o fragmento de documento. '
    "El modelo analizará su contenido y estimará el ODS con mayor relación semántica.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Ejemplos")
    selected_example = st.selectbox(
        "Selecciona un tema",
        ["— Elegir —", *EXAMPLES],
        label_visibility="collapsed",
    )
    if st.button("Usar este ejemplo", use_container_width=True, disabled=selected_example == "— Elegir —"):
        st.session_state.input_text = EXAMPLES[selected_example]
        st.rerun()

    st.divider()
    st.markdown("**Acerca del modelo**")
    st.caption(
        "Pipeline completo con TF-IDF, reducción dimensional mediante SVD y "
        "regresión logística. Fue entrenado con las 16 clases presentes en el "
        "conjunto de datos (ODS 1 a 16)."
    )

with st.form("classification_form"):
    text = st.text_area(
        "Texto para analizar",
        key="input_text",
        height=180,
        max_chars=5000,
        placeholder="Ejemplo: El programa instalará sistemas de energía solar en comunidades rurales…",
        help="Puedes ingresar hasta 5.000 caracteres.",
    )
    submitted = st.form_submit_button("Analizar texto", type="primary", use_container_width=True)

if submitted:
    clean_text = " ".join(text.split())
    if len(clean_text) < 20:
        st.warning("Escribe un texto un poco más descriptivo (al menos 20 caracteres).")
    else:
        try:
            ods_number, confidence, ranking = predict(clean_text)
        except Exception as error:
            st.error("No fue posible cargar el modelo o generar la predicción.")
            with st.expander("Detalle técnico"):
                st.code(f"{type(error).__name__}: {error}")
        else:
            st.divider()
            image_column, result_column = st.columns([1, 3], vertical_alignment="center")
            with image_column:
                icon_path = ICONS_DIR / ODS[ods_number][1]
                if icon_path.is_file():
                    st.image(str(icon_path), width=180)
            with result_column:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Predicción principal</div>
                        <div class="result-title">ODS {ods_number} · {ODS[ods_number][0]}</div>
                        <div class="result-confidence">Confianza estimada: {confidence:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.subheader("Otras categorías consideradas")
            st.bar_chart(
                ranking,
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

