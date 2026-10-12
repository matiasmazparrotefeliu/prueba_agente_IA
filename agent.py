from datetime import date
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

from config import API_URL, API_KEY, AI_MODEL

# Límite determinista de búsquedas por pregunta. Corta el loop sin depender
# de que el modelo free "decida" parar (máximo 3 idas a tools por turno).
MAX_TOOL_CALLS = 3

# 1. Definir el Estado Compartido (Memoria de trabajo del agente)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    # Contador de cuántas veces hemos pasado por el nodo tools en ESTE turno.
    # Sin reducer: cada nodo lo sobreescribe con el valor calculado.
    tool_iterations: int

# 2. Configurar el LLM apuntando a OpenRouter
# Usamos ChatOpenAI ya que OpenRouter es compatible con la API de OpenAI.
# Lección 1: límites de seguridad para modelos free (con cuota y fallos):
# - max_retries: reintenta 2 veces ante errores transitorios (429/500).
# - request_timeout: no quedarse colgado más de 60s por llamada.
# - max_tokens: acota el costo y evita respuestas eternas.
llm = ChatOpenAI(
    base_url=API_URL,
    api_key=API_KEY,
    model=AI_MODEL,
    temperature=0.1,
    max_retries=2,
    request_timeout=60,
    # max_tokens: nemotron es reasoning-model; sus reasoning tokens consumen
    # este presupuesto. Con 4000 el modelo razona y llega truncado (content='').
    # 8000 da espacio a reasoning + respuesta completa.
    # NOTA: no pasar reasoning_effort via model_kwargs (ChatOpenAI nuevo lo
    # exige como parámetro explícito y genera UserWarning). Se deja fuera
    # para máxima compatibilidad entre versiones local/Docker.
    max_tokens=8000,
)

# 3. Definir las Herramientas
# _search_backend es el motor real. Se envuelve con @tool para exponer al LLM
# un nombre en español ("buscar_web") con type hints limpios (str -> str).
# NO usar Tool(func=search_tool.run): sus anotaciones internas tienen un
# forward-ref a 'uuid' que rompe ToolNode/get_type_hints (NameError en Docker).
_search_backend = DuckDuckGoSearchRun()


@tool("buscar_web")
def buscar_web(query: str) -> str:
    """Busca información actual en internet. Úsala para datos frescos, noticias, alineaciones, resultados, hechos que no conoces."""
    return _search_backend.run(query)


tools = [buscar_web]

# Conectar las herramientas al LLM para que sepa que puede usarlas
llm_with_tools = llm.bind_tools(tools)

# 4. Definir los Nodos del Grafo

# Nodo del Agente (El Cerebro)
# Lección 1: system prompt. Sin esto el modelo no sabe idioma, rol ni
# cuándo buscar. Lo inyectamos solo para esta llamada (no lo guardamos
# en el estado) para no duplicarlo en el historial del checkpointer.
def call_model(state: AgentState):
    today = date.today().isoformat()
    system = SystemMessage(
        content=(
            "Eres un asistente de investigación. Respondes SIEMPRE en español. "
            f"Fecha actual: {today} (úsala SOLO para contextualizar 'hoy', 'esta semana', 'próximo partido'). "
            "NO inventes fechas de eventos futuros ni alineaciones no confirmadas. "
            "REGLAS CRÍTICAS: "
            "1) NUNCA escribas tu proceso de pensamiento, razonamiento, chain-of-thought, planes, ni 'voy a buscar'. Solo respuesta final. "
            "2) Si el usuario saluda o pregunta algo conversacional → responde 1 frase natural. NO uses tools. "
            "3) Para alineaciones/partidos: SIEMPRE busca 'próximo partido [equipo1] [equipo2]' SIN fecha. "
            "   Si no hay info oficial → 'No hay alineación confirmada aún'. "
            "4) La herramienta se llama 'buscar_web'. Úsala exactamente con ese nombre (máx 2 búsquedas/pregunta). "
            "5) Cita fuente breve: (fuente: diario X, web Y). "
            "Sé conciso, directo, cero preámbulos."
        )
    )
    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Nodo de Herramientas (Las Manos)
# ToolNode de LangGraph ejecuta automáticamente la herramienta solicitada por el LLM.
# handle_tool_errors=True devuelve el error al LLM como mensaje de herramienta
# en vez de romper todo el grafo (clave con DDGS y rate-limits).
tool_node = ToolNode(tools=tools, handle_tool_errors=True)

# Nodo "tools" con contador: envolvemos ToolNode para incrementar
# tool_iterations cada vez que se ejecuta una búsqueda en el turno actual.
def call_tools(state: AgentState):
    result = tool_node.invoke(state)
    result["tool_iterations"] = state.get("tool_iterations", 0) + 1
    return result

# Nodo de resumen forzado: si el agente agotó sus búsquedas y sigue sin
# respuesta, aquí le pedimos que cierre CON LO QUE YA TIENE (sin tools).
# Usa llm "pelón" (sin bind_tools) para que sea imposible que vuelva a buscar.
def summary_node(state: AgentState):
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Ya no hay más búsquedas disponibles. Con la información que "
                    "tienes en la conversación, da tu mejor respuesta final ahora. "
                    "Si no tienes el dato, dilo honestamente."
                )
            )
        ]
        + state["messages"]
    )
    return {"messages": [response]}

# 5. Definir la Arista Condicional (Enrutador)
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Si el LLM emitió una llamada a herramienta:
    if last_message.tool_calls:
        # ¿Ya agotamos las búsquedas permitidas en este turno?
        if state.get("tool_iterations", 0) >= MAX_TOOL_CALLS:
            return "summary"  # cortamos el loop y forzamos respuesta final
        return "tools"
    
    # Si no, terminamos el ciclo y respondemos al usuario
    return END

# 6. Construir el Grafo con LangGraph
workflow = StateGraph(AgentState)

# Agregar los nodos
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)
workflow.add_node("summary", summary_node)

# Establecer el punto de entrada
workflow.set_entry_point("agent")

# Agregar aristas condicionales y ciclos
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "summary": "summary",
        END: END
    }
)

# Desde el nodo de herramientas, siempre volvemos al agente para que procese el resultado
workflow.add_edge("tools", "agent")

# El resumen forzado siempre termina el ciclo
workflow.add_edge("summary", END)

# Creamos un archivo local llamado 'checkpoints.db' para guardar el historial
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Compilar el grafo para que esté listo para usarse
graph = workflow.compile(checkpointer=memory)
