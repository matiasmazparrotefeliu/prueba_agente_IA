from agent import graph

def main():
    print("==================================================")
    print("🤖 ¡Asistente de Investigación IA Iniciado!")
    print(f"Modelo en uso: {graph}")
    print("Escribe 'salir' o 'exit' para terminar la conversación.")
    print("==================================================\n")

    # Definimos un identificador único para esta sesión de chat
    config = {"configurable": {"thread_id": "sesion_usuario_1"}}

    while True:
        try:
            user_input = input("\nTú: ")
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("¡Adiós!")
                break
            
            if not user_input.strip():
                continue

            # Estructurar la entrada para LangGraph
            inputs = {"messages": [("user", user_input)]}

            print("\nAI: Pensando y buscando información si es necesario...")
            
            # Ejecutamos el grafo pasándole tanto las entradas como la configuración con el thread_id
            response = graph.invoke(inputs, config=config)

            # Obtener el último mensaje generado por el agente
            latest_message = response["messages"][-1]
            print(f"\nAI: {latest_message.content}")

        except Exception as e:
            print(f"\n❌ Ocurrió un error: {e}")

if __name__ == "__main__":
    main()
