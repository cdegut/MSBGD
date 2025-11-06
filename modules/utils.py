import dearpygui.dearpygui as dpg

log_string = ""


def log(message: str) -> None:
    global log_string
    log_string += message + "\n"
    dpg.set_value("message_box", log_string)
    print(message)
