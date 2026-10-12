import sys
import time
import threading
from uuid import uuid4

from langgraph.errors import GraphRecursionError

from agent import graph
from config import AI_MODEL

MAX_STEPS = 25

class Spinner:
    def __init__(self, message="Procesando"):
        self.message = message
        self.running = False
        self.thread = None
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
    
    def _spin(self):
        i = 0
        while self.running:
            sys.stdout.write(f"\rAI: {self.frames[i]} {self.message}...")
            sys.stdout.flush()
            time.sleep(0.08)
            i = (i + 1) % len(self.frames)
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

def main():
    print("==================================================")
    print("🤖 ¡Asistente de Investigación IA Iniciado!")
    print(f"Modelo en uso: {AI_MODEL}")
    print("Comandos: /nuevo (reiniciar sesión), salir/exit/quit")
    print("==================================================\n")

    thread_id = str(uuid4())

    while True:
        try:
            user_input = input("\nTú: ")
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("¡Adiós!")
                break

            if user_input.lower() == "/nuevo":
                thread_id = str(uuid4())
                print("🧹 Sesión nueva iniciada. Memoria limpia.")
                continue

            if not user_input.strip():
                continue

            inputs = {"messages": [("user", user_input)], "tool_iterations": 0}

            spinner = Spinner("Consultando")
            spinner.start()
            try:
                response = graph.invoke(
                    inputs,
                    {
                        "configurable": {"thread_id": thread_id},
                        "recursion_limit": MAX_STEPS,
                    },
                )
            finally:
                spinner.stop()

            latest_message = response["messages"][-1]
            if latest_message.content:
                print(f"\nAI: {latest_message.content}")
            else:
                print("\nAI: ⚠️ Respuesta vacía del modelo. "
                      "Prueba de nuevo o usa /nuevo.")

        except GraphRecursionError:
            print("\n🛑 El agente entró en bucle y frené por seguridad.")
            print("   Usa /nuevo para reiniciar la sesión y prueba otra pregunta.")

        except Exception as e:
            print(f"\n❌ Ocurrió un error: {e}")

if __name__ == "__main__":
    main()
