import io
import streamlit as st
import pandas as pd
import hashlib
from datetime import datetime

from chatbot.context_builder import build_user_context, build_multi_user_context
from chatbot.llm_client import ask_groq, transcribir_audio
from chatbot.prompts import SYSTEM_PROMPT


# ─────────────────────────────────────────────────────────────
# UTILIDADES COMUNES
# ─────────────────────────────────────────────────────────────

def _hora_a_periodo_label(hora: int) -> str:
    if 6 <= hora < 13:
        return "mañana"
    elif 13 <= hora < 20:
        return "tarde"
    elif 20 <= hora < 24:
        return "noche"
    else:
        return "madrugada"


def _get_canciones_escuchadas(data: dict, persona_id: str) -> set:
    canciones = set()
    if "usuario_track" in data:
        df = data["usuario_track"]
        user_df = df[df["persona_id"] == persona_id]
        if "nombre_cancion" in user_df.columns:
            canciones.update(user_df["nombre_cancion"].dropna().str.lower().tolist())
    return canciones


def _get_artistas_escuchados(data: dict, persona_id: str) -> set:
    artistas = set()
    if "artistas" in data:
        df = data["artistas"]
        user_df = df[df["persona_id"] == persona_id]
        if "nombre_artista" in user_df.columns:
            artistas.update(user_df["nombre_artista"].dropna().str.lower().tolist())
    return artistas


def _enviar_mensaje(system_completo: str, user_input: str):
    """Llama al LLM y gestiona historial. Devuelve la respuesta."""
    historial_api = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history
    ]
    try:
        respuesta = ask_groq(system_completo, user_input, historial_api)
    except Exception as e:
        respuesta = f"Lo siento, ha ocurrido un error al conectar con el asistente: {e}"

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": respuesta})
    return respuesta


# ─────────────────────────────────────────────────────────────
# CARGA DE LA TABLA GLOBAL DE USUARIOS (para modo grupal)
# ─────────────────────────────────────────────────────────────

@st.cache_data
def _cargar_tabla_usuarios() -> pd.DataFrame:
    """
    Carga la tabla completa usuario_track desde data/.
    Se cachea para no leer el CSV en cada interacción.
    """
    try:
        df = pd.read_csv("data/usuario_track.csv")
        df = df[~df["persona_id"].str.contains(",", na=False)]
        df = df.dropna(subset=["persona_id", "nombre_cancion", "nombre_artista"])
        df["persona_id"] = df["persona_id"].str.strip()
        # Normalizar nombre_cancion y nombre_artista para comparaciones
        df["_cancion_norm"] = df["nombre_cancion"].str.lower().str.strip()
        df["_artista_norm"] = df["nombre_artista"].str.lower().str.strip()
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def _get_todos_usuarios(df_global: pd.DataFrame) -> list:
    if df_global.empty or "persona_id" not in df_global.columns:
        return []
    return sorted(df_global["persona_id"].unique().tolist())


# ─────────────────────────────────────────────────────────────
# LÓGICA DE RECOMENDACIÓN GRUPAL
# ─────────────────────────────────────────────────────────────

def _construir_contexto_grupal(df_global: pd.DataFrame, personas: list) -> str:
    if df_global.empty:
        return "No hay datos disponibles de los usuarios seleccionados."

    partes = []

    for persona in personas:
        df_p = df_global[df_global["persona_id"] == persona].copy()
        if df_p.empty:
            partes.append(f"\n[{persona.upper()}]: Sin datos disponibles.")
            continue

        top_canciones = (df_p.sort_values("score_interes", ascending=False)
                         .head(15)[["nombre_cancion", "nombre_artista", "score_interes"]])

        top_artistas = (df_p.groupby("nombre_artista")
                        .agg(total_min=("minutos_totales", "sum"),
                             n_canciones=("nombre_cancion", "nunique"))
                        .sort_values("total_min", ascending=False)
                        .head(10)
                        .reset_index())

        canciones_txt = "\n".join(
            f"  - {r['nombre_cancion']} ({r['nombre_artista']})"
            for _, r in top_canciones.iterrows()
        )
        artistas_txt = "\n".join(
            f"  - {r['nombre_artista']} ({int(r['total_min'])} min, {int(r['n_canciones'])} canciones)"
            for _, r in top_artistas.iterrows()
        )

        partes.append(
            f"\n{'='*45}\n"
            f"PERFIL DE {persona.upper()}\n"
            f"Total canciones escuchadas: {len(df_p)}\n"
            f"Top canciones por interés:\n{canciones_txt}\n"
            f"Top artistas por minutos:\n{artistas_txt}"
        )

    # ── Canciones comunes: comparar por cancion+artista normalizados ──
    # FIX: antes comparábamos solo por nombre_cancion en minúsculas,
    # lo que podía dar falsos positivos (canciones con el mismo nombre
    # pero de artistas distintos). Ahora cruzamos cancion+artista juntos.
    sets_pares = [
        set(
            zip(
                df_global[df_global["persona_id"] == p]["_cancion_norm"],
                df_global[df_global["persona_id"] == p]["_artista_norm"]
            )
        )
        for p in personas
    ]
    pares_comunes = sets_pares[0]
    for s in sets_pares[1:]:
        pares_comunes = pares_comunes & s

    if pares_comunes:
        lista = "\n".join(f"  - {c} — {a}" for c, a in sorted(pares_comunes)[:20])
        partes.append(
            f"\n{'='*45}\n"
            f"CANCIONES QUE TODOS CONOCEN ({len(pares_comunes)} en común):\n{lista}"
        )
    else:
        partes.append(
            f"\n{'='*45}\n"
            "CANCIONES EN COMÚN: Ninguna canción exacta en común entre todos los usuarios."
        )

    # ── Artistas comunes ────────────────────────────────────────
    sets_artistas = [
        set(df_global[df_global["persona_id"] == p]["_artista_norm"])
        for p in personas
    ]
    artistas_comunes = sets_artistas[0]
    for s in sets_artistas[1:]:
        artistas_comunes = artistas_comunes & s

    if artistas_comunes:
        partes.append(
            "ARTISTAS QUE TODOS ESCUCHAN:\n"
            + "\n".join(f"  - {a}" for a in sorted(artistas_comunes)[:15])
        )

    # ── Canciones que uno conoce y el otro no (solo para grupos de 2) ──
    if len(personas) == 2:
        p1, p2 = personas[0], personas[1]
        pares_p1 = sets_pares[0]
        pares_p2 = sets_pares[1]

        solo_p1 = pares_p1 - pares_p2
        solo_p2 = pares_p2 - pares_p1

        if solo_p1:
            df_p1 = df_global[df_global["persona_id"] == p1]
            top = (df_p1[
                df_p1["_cancion_norm"].isin([c for c, _ in solo_p1]) &
                df_p1["_artista_norm"].isin([a for _, a in solo_p1])
            ].sort_values("score_interes", ascending=False).head(8))
            txt = "\n".join(f"  - {r['nombre_cancion']} ({r['nombre_artista']})"
                            for _, r in top.iterrows())
            partes.append(f"TOP CANCIONES DE {p1.upper()} QUE {p2.upper()} AÚN NO CONOCE:\n{txt}")

        if solo_p2:
            df_p2 = df_global[df_global["persona_id"] == p2]
            top = (df_p2[
                df_p2["_cancion_norm"].isin([c for c, _ in solo_p2]) &
                df_p2["_artista_norm"].isin([a for _, a in solo_p2])
            ].sort_values("score_interes", ascending=False).head(8))
            txt = "\n".join(f"  - {r['nombre_cancion']} ({r['nombre_artista']})"
                            for _, r in top.iterrows())
            partes.append(f"TOP CANCIONES DE {p2.upper()} QUE {p1.upper()} AÚN NO CONOCE:\n{txt}")

    partes.append(f"{'='*45}")
    return "\n\n".join(partes)


def _cancion_que_mas_une(df_global: pd.DataFrame, personas: list) -> pd.DataFrame:
    """
    Canciones escuchadas por más de un miembro del grupo,
    cruzando por cancion+artista normalizados para evitar falsos positivos.
    """
    df_sel = df_global[df_global["persona_id"].isin(personas)].copy()
    agg = (df_sel.groupby(["_cancion_norm", "_artista_norm"])
           .agg(
               nombre_cancion=("nombre_cancion", "first"),
               nombre_artista=("nombre_artista", "first"),
               n_usuarios=("persona_id", "nunique"),
               score_medio=("score_interes", "mean"),
           )
           .reset_index(drop=True)
           .sort_values(["n_usuarios", "score_medio"], ascending=[False, False])
           .head(10))
    return agg


# ─────────────────────────────────────────────────────────────
# SUGERENCIAS DE PREGUNTAS
# ─────────────────────────────────────────────────────────────

SUGERENCIAS_INDIVIDUAL = [
    "🎵 Recomiéndame canciones para estudiar",
    "🎶 Recomiéndame canciones parecidas a La Macarena",
    "🎤 Recomiéndame canciones del artista Beethoven",
    "🌙 ¿Qué debería escuchar esta noche?",
    "⚡ Dame música energética para entrenar",
]

SUGERENCIAS_GRUPAL = [
    "🎧 Danos una playlist de mínimo 10 canciones que nos gusten a todos",
    "❤️ ¿Cuál es la canción que más nos une?",
    "🔀 Recomiéndame canciones que escucha mi amigo/a y que yo no conozca",
    "🎯 ¿En qué géneros o estilos coincidimos más?",
    "🎉 Haz una playlist perfecta para una fiesta con los dos",
]


def _chips_sugerencias(sugerencias: list, key_prefix: str):
    cols = st.columns(len(sugerencias))
    for i, (col, texto) in enumerate(zip(cols, sugerencias)):
        with col:
            if st.button(texto, key=f"{key_prefix}_sug_{i}", use_container_width=True):
                st.session_state[f"{key_prefix}_sugerencia_activa"] = texto


# ─────────────────────────────────────────────────────────────
# VOZ: botón al lado del chat_input mediante HTML/JS
# ─────────────────────────────────────────────────────────────

def _render_voz(key_prefix: str):
    """
    Muestra st.audio_input con el estado visible:
    - Mientras graba: spinner "Grabando..."
    - Cuando hay audio: transcribe y mete el texto en la caja de chat
    El widget se coloca encima del chat_input con una etiqueta clara.
    """
    # Estado del grabador
    grabando_key = f"{key_prefix}_grabando"

    col_label, col_estado = st.columns([1, 3])
    with col_label:
        st.markdown("**🎙️ Voz:**")
    with col_estado:
        if st.session_state.get(grabando_key):
            st.markdown("🔴 *Grabando... pulsa Stop cuando termines*")
        else:
            st.markdown("*Pulsa el micro para grabar*")

    # Usamos una key dinámica para resetear el widget tras cada transcripción
    audio_key_counter = st.session_state.get(f"{key_prefix}_audio_key_counter", 0)

    audio = st.audio_input(
        "Grabar mensaje de voz",
        key=f"{key_prefix}_audio_input_{audio_key_counter}",
        label_visibility="collapsed",
    )

    if audio is not None:
        # Marcar que hay grabación en curso al principio del ciclo
        st.session_state[grabando_key] = False

        try:
            audio_bytes = audio.getvalue()
        except AttributeError:
            audio.seek(0)
            audio_bytes = audio.read()

        if len(audio_bytes) < 500:
            st.caption("⚠️ Audio muy corto, inténtalo de nuevo.")
            return

        # Hash para no retranscribir el mismo audio
        nuevo_hash = hashlib.md5(audio_bytes).hexdigest()
        ultimo_hash = st.session_state.get(f"{key_prefix}_ultimo_audio_hash")

        if nuevo_hash == ultimo_hash:
            return  # ya procesado, no hacer nada

        st.session_state[f"{key_prefix}_ultimo_audio_hash"] = nuevo_hash

        with st.spinner("✍️ Transcribiendo tu mensaje..."):
            texto = transcribir_audio(audio_bytes)

        if texto and texto.strip():
            # Resetear el widget incrementando la key (borra visualmente el estado de error)
            st.session_state[f"{key_prefix}_audio_key_counter"] = audio_key_counter + 1
            # Inyectar como si el usuario lo hubiera escrito
            st.session_state[f"{key_prefix}_sugerencia_activa"] = texto.strip()
            st.rerun()
        else:
            st.warning("No se pudo transcribir. Habla más cerca del micrófono e inténtalo de nuevo.")
    else:
        # No hay audio aún: marcar estado "esperando grabación"
        st.session_state[grabando_key] = False


# ─────────────────────────────────────────────────────────────
# MODO INDIVIDUAL
# ─────────────────────────────────────────────────────────────

def _render_individual(data: dict, persona_id: str, ahora: datetime, dia_semana: str):
    st.markdown("#### 💡 Sugerencias de preguntas")
    _chips_sugerencias(SUGERENCIAS_INDIVIDUAL, key_prefix="ind")

    st.divider()

    for mensaje in st.session_state.chat_history:
        with st.chat_message(mensaje["role"]):
            st.write(mensaje["content"])

    sugerencia = st.session_state.pop("ind_sugerencia_activa", None)

    # Micrófono encima del chat_input
    with st.container():
        _render_voz(key_prefix="ind")

    user_input = st.chat_input("¿Qué quieres escuchar ahora? ¿Estoy en bucle con una canción, recomiéndame algo parecido...")

    mensaje_a_enviar = sugerencia or user_input

    if mensaje_a_enviar:
        perfil_context = build_user_context(
            data, persona_id,
            hora_actual=ahora.hour,
            dia_semana=dia_semana
        )
        cabecera = (
            f"Modo: recomendación individual\n"
            f"Usuario: {persona_id}\n"
            f"Hora actual: {ahora.strftime('%H:%M')} ({dia_semana})"
        )

        canciones_escuchadas = _get_canciones_escuchadas(data, persona_id)
        artistas_escuchados = _get_artistas_escuchados(data, persona_id)

        if canciones_escuchadas:
            lista = ", ".join(sorted(canciones_escuchadas)[:80])
            cabecera += f"\n\nCANCIONES QUE YA CONOCE (NO recomendar estas): {lista}"

        if artistas_escuchados:
            lista = ", ".join(sorted(artistas_escuchados)[:50])
            cabecera += f"\n\nARTISTAS QUE YA ESCUCHA (prioriza artistas NUEVOS): {lista}"

        system_completo = f"{SYSTEM_PROMPT}\n\n--- CONTEXTO ACTUAL ---\n{cabecera}\n\n{perfil_context}"

        with st.chat_message("user"):
            st.write(mensaje_a_enviar)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                respuesta = _enviar_mensaje(system_completo, mensaje_a_enviar)
            st.write(respuesta)

        st.rerun()


# ─────────────────────────────────────────────────────────────
# MODO GRUPAL
# ─────────────────────────────────────────────────────────────

def _render_grupal(data: dict, persona_id: str, ahora: datetime, dia_semana: str):

    df_global = _cargar_tabla_usuarios()
    todos_usuarios = _get_todos_usuarios(df_global)

    if not todos_usuarios:
        st.error(
            "No se encontró `data/usuario_track.csv`. "
            "Asegúrate de que el fichero existe en la carpeta `data/`."
        )
        return

    # ── CAMBIO 1: buscador + selector integrado en un solo widget ──
    st.markdown("### 👥 Añadir amigos al grupo")

    # Excluir al usuario actual
    otros_usuarios = [u for u in todos_usuarios if u != persona_id]

    amigos_seleccionados = st.multiselect(
    "🔍 Buscar y seleccionar amigos",
    options=otros_usuarios,
    default=st.session_state.get("amigos_seleccionados_prev", []),
    placeholder="Escribe un nombre y pulsa Enter para seleccionarlo...",
    help="Puedes escribir para filtrar y seleccionar uno o varios amigos")

    # Limpiar historial si cambian los amigos seleccionados
    amigos_prev = st.session_state.get("amigos_seleccionados_prev")
    if amigos_prev is not None and set(amigos_prev) != set(amigos_seleccionados):
        st.session_state.chat_history = []
    st.session_state["amigos_seleccionados_prev"] = amigos_seleccionados

    grupo = [persona_id] + amigos_seleccionados

    if len(grupo) < 2:
        st.info("👆 Selecciona al menos un amigo para empezar.")
        return

    # ── Panel de afinidad ──────────────────────────────────────
    st.markdown(f"### 🎵 Grupo: {' · '.join(grupo)}")

    with st.expander("🔗 Ver canciones que os unen", expanded=False):
        df_une = _cancion_que_mas_une(df_global, grupo)
        df_une_comunes = df_une[df_une["n_usuarios"] >= 2]
        if df_une_comunes.empty:
            st.info("No hay canciones que todos hayáis escuchado. ¡Quizás sea el momento de descubrirlas juntos!")
        else:
            st.markdown(f"**{len(df_une_comunes)} canciones en común:**")
            for _, r in df_une_comunes.iterrows():
                st.markdown(
                    f"🎵 **{r['nombre_cancion']}** — {r['nombre_artista']} "
                    f"&nbsp;|&nbsp; {int(r['n_usuarios'])}/{len(grupo)} usuarios"
                )

    # ── Sugerencias ────────────────────────────────────────────
    st.markdown("#### 💡 Sugerencias de preguntas grupales")
    _chips_sugerencias(SUGERENCIAS_GRUPAL, key_prefix="grp")

    st.divider()

    # ── Historial ──────────────────────────────────────────────
    for mensaje in st.session_state.chat_history:
        with st.chat_message(mensaje["role"]):
            st.write(mensaje["content"])

    sugerencia = st.session_state.pop("grp_sugerencia_activa", None)

    # Micrófono encima del chat_input
    with st.container():
        _render_voz(key_prefix="grp")

    user_input = st.chat_input("Pregunta algo sobre el grupo...")
    mensaje_a_enviar = sugerencia or user_input

    if mensaje_a_enviar:
        contexto_grupal = _construir_contexto_grupal(df_global, grupo)
        cabecera = (
            f"Modo: recomendación grupal\n"
            f"Usuarios del grupo: {', '.join(grupo)}\n"
            f"Hora actual: {ahora.strftime('%H:%M')} ({dia_semana})\n\n"
            f"INSTRUCCIÓN ESPECIAL: Recomienda únicamente artistas y canciones que NINGUNO "
            f"de los usuarios del grupo haya escuchado ya. "
            f"Usa los datos de afinidad para justificar las recomendaciones."
        )

        system_completo = (
            f"{SYSTEM_PROMPT}\n\n"
            f"--- CONTEXTO DEL GRUPO ---\n{cabecera}\n\n"
            f"{contexto_grupal}"
        )

        with st.chat_message("user"):
            st.write(mensaje_a_enviar)

        with st.chat_message("assistant"):
            with st.spinner("Analizando gustos del grupo..."):
                respuesta = _enviar_mensaje(system_completo, mensaje_a_enviar)
            st.write(respuesta)

        st.rerun()


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

def render_chatbot(data: dict, persona_id: str):
    st.markdown("""
    <style>
    /* Oculta contenedor de error externo por si acaso */
    [data-testid="stAudioInput"] ~ div[role="alert"],
    [data-testid="stAudioInput"] ~ p {
        display: none !important;
    }
    </style>

    <script>
    (function() {
        function patchAudioWidget() {
            // Buscar todos los elementos con shadow root (web components del audio_input)
            const allElements = document.querySelectorAll('*');
            allElements.forEach(el => {
                if (el.shadowRoot) {
                    try {
                        // Inyectar CSS dentro del shadow DOM para ocultar el mensaje de error
                        const style = document.createElement('style');
                        style.textContent = `
                            [class*="error"], [class*="Error"],
                            p, span[class*="status"],
                            div[class*="message"] {
                                display: none !important;
                            }
                        `;
                        if (!el.shadowRoot.querySelector('style[data-patched]')) {
                            style.setAttribute('data-patched', 'true');
                            el.shadowRoot.appendChild(style);
                        }

                        // Reemplazar texto del error directamente si aparece
                        const walker = document.createTreeWalker(
                            el.shadowRoot,
                            NodeFilter.SHOW_TEXT
                        );
                        let node;
                        while ((node = walker.nextNode())) {
                            if (node.textContent.includes('An error has occurred')) {
                                node.textContent = '';
                            }
                        }
                    } catch(e) {}
                }
            });
        }

        // Ejecutar al cargar y repetir cada vez que el DOM cambie
        const observer = new MutationObserver(patchAudioWidget);
        observer.observe(document.body, { childList: true, subtree: true });
        patchAudioWidget();
    })();
    </script>
    """, unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    ahora = datetime.now()
    dias_es = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    dia_semana = dias_es[ahora.weekday()]

    # ── Cabecera: modo + contexto + limpiar + invitar ───────────
    col1, col2, col3 = st.columns([3, 2, 1])

    with col1:
        modo = st.radio(
            "Modo",
            ["🎧 Individual", "👥 Grupal"],
            horizontal=True,
            help="Individual: recomendaciones solo para ti. Grupal: playlist para el grupo."
        )

    # Limpiar historial automáticamente al cambiar de modo
    modo_anterior = st.session_state.get("modo_anterior")
    if modo_anterior is not None and modo_anterior != modo:
        st.session_state.chat_history = []
        st.session_state.pop("amigos_seleccionados_prev", None)
    st.session_state["modo_anterior"] = modo
    with col2:
        with st.expander("ℹ️ Contexto actual"):
            st.write(f"**Usuario:** {persona_id}")
            st.write(f"**Hora:** {ahora.strftime('%H:%M')} ({_hora_a_periodo_label(ahora.hour)})")
            st.write(f"**Día:** {dia_semana.capitalize()}")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        # CAMBIO 3 (limpiar chat)
        if st.button("🗑️ Limpiar chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.pop("amigos_seleccionados_prev", None)
            st.rerun()

        # CAMBIO 2: botón invitar debajo de limpiar chat
        if st.button("📨 Invitar amigo", use_container_width=True, key="btn_invitar_cabecera"):
            st.session_state.mostrar_invitar = not st.session_state.get("mostrar_invitar", False)

    # Panel de invitación (se muestra bajo la cabecera si está activo)
    if st.session_state.get("mostrar_invitar", False):
        with st.container(border=True):
            st.markdown("#### 📨 Invita a un amigo")
            st.markdown(
                "Para unirse, tu amigo necesita descargar sus datos de Spotify y subirlos a la app."
            )
            st.markdown(
                "**Paso 1:** Ir a la cuenta de Spotify y solicitar los datos:\n\n"
                "👉 [Descargar datos de Spotify](https://www.spotify.com/es/account/privacy/)"
            )
            st.markdown(
                "**Paso 2:** Una vez descargados (~30 días de espera), abrir esta app, "
                "introducir el nombre y subir el ZIP."
            )
            st.info("💡 Cuando suba sus datos aparecerá automáticamente en el buscador.")
            if st.button("✕ Cerrar", key="btn_cerrar_invitar"):
                st.session_state.mostrar_invitar = False
                st.rerun()

    st.divider()

    if modo == "🎧 Individual":
        _render_individual(data, persona_id, ahora, dia_semana)
    else:
        _render_grupal(data, persona_id, ahora, dia_semana)
