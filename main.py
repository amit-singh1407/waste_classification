import os

from ui_final import create_app, get_launch_host, get_launch_port

app = create_app()
demo = app


if __name__ == "__main__":
    app.launch(server_name=get_launch_host(), server_port=get_launch_port())

