import pandas as pd
import dearpygui.dearpygui as dpg

file_path = rf"D:\MassSpec\Um data.xlsx"


xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

def load_sheet(sender, app_data, user_data):
    selected_sheet = dpg.get_value("sheet_selector")
    df = pd.read_excel(file_path, sheet_name=selected_sheet)
    print(df)

dpg.create_context()

with dpg.window(label="Excel Sheet Selector"):
    dpg.add_text("Select a sheet:")
    dpg.add_combo(sheet_names, tag="sheet_selector")
    dpg.add_button(label="Load Sheet", callback=load_sheet)

dpg.create_viewport(title='Excel Sheet Selector', width=600, height=200)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()