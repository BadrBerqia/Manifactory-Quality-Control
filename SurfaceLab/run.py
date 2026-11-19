"""Démarre SurfaceLab en local et ouvre l'adresse quand le serveur répond."""

import argparse
import socket
import threading
import time
import urllib.error
import urllib.request
import webbrowser

import uvicorn


def open_when_ready(url, server):
    while not server.should_exit:
        try:
            with urllib.request.urlopen(url + "/api/catalog", timeout=1):
                webbrowser.open(url)
                return
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="0 choisit un port libre")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    # Garder la socket jusqu'au démarrage évite la concurrence pour le port.
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", args.port))
        except OSError as error:
            if args.port != 8765:
                parser.error(str(error))
            listener.bind(("127.0.0.1", 0))
        address, port = listener.getsockname()
        url = f"http://{address}:{port}"
        config = uvicorn.Config("serve:app", host=address, port=port, access_log=False)
        server = uvicorn.Server(config)
        print(f"SurfaceLab — {url}", flush=True)
        if not args.no_browser:
            threading.Thread(target=open_when_ready, args=(url, server), daemon=True).start()
        server.run(sockets=[listener])


if __name__ == "__main__":
    main()
