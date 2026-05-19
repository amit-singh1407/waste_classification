from app import demo, get_launch_host, get_launch_port

app = demo


if __name__ == "__main__":
    app.launch(server_name=get_launch_host(), server_port=get_launch_port())

