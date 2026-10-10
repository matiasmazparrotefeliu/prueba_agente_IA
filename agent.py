import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
AI_MODEL = os.getenv("AI_MODEL")

# 1. Definir el Estado Compartido (Memoria de trabajo del agente)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 2. Configurar el LLM apuntando a OpenRouter
# Usamos ChatOpenAI ya que OpenRouter es compatible con la API de OpenAI
llm = ChatOpenAI(
    base_url=API_URL,
    api_key=API_KEY,
    model=AI_MODEL,
    temperature=0.1
)

# 3. Definir las Herramientas
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# Conectar las herramientas al LLM para que sepa que puede usarlas
llm_with_tools = llm.bind_tools(tools)

# 4. Definir los Nodos del Grafo

# Nodo del Agente (El Cerebro)
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Nodo de Herramientas (Las Manos)
# ToolNode de LangGraph ejecuta automáticamente la herramienta solicitada por el LLM
tool_node = ToolNode(tools=tools)

# 5. Definir la Arista Condicional (Enrutador)
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Si el LLM emitió una llamada a herramienta, vamos al nodo "tools"
    if last_message.tool_calls:
        return "tools"
    
    # Si no, terminamos el ciclo y respondemos al usuario
    return END

# 6. Construir el Grafo con LangGraph
workflow = StateGraph(AgentState)

# Agregar los nodos
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Establecer el punto de entrada
workflow.set_entry_point("agent")

# Agregar aristas condicionales y ciclos
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# Desde el nodo de herramientas, siempre volvemos al agente para que procese el resultado
workflow.add_edge("tools", "agent")

# Creamos un archivo local llamado 'checkpoints.db' para guardar el historial
conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Compilar el grafo para que esté listo para usarse
graph = workflow.compile(checkpointer=memory)
