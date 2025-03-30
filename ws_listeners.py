import asyncio
import websockets

async def listen():
    # Change this URI to your actual WebSocket endpoint.
    uri = "ws://localhost:8000/ws/entities/"
    async with websockets.connect(uri) as websocket:
        print("Connected to", uri)
        while True:
            try:
                message = await websocket.recv()
                print("Received:", message)
            except websockets.ConnectionClosed:
                print("Connection closed")
                break

if __name__ == "__main__":
    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        print("Exiting...")