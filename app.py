import subprocess
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage:\n  python app.py ingest\n  python app.py chat")
        return

    cmd = sys.argv[1].lower()
    if cmd == "ingest":
        subprocess.run([sys.executable, "ingest.py"], check=True)
    elif cmd == "chat":
        subprocess.run([sys.executable, "rag_agent.py"], check=True)
    else:
        print("Unknown command:", cmd)

if __name__ == "__main__":
    main()
