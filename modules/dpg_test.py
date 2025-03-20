import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title='Custom Title', width=900, height=800)
window = dpg.window( label="------", width=900, height=800, tag="Primary Window" )

with window as main_win:
	with dpg.tab_bar(label="Test", tag='tab_bar', pos=[500, 0]): # <- Not working
		with dpg.tab(label="Test Setup"):
			dpg.add_text('Testing 1')
		with dpg.tab(label="Monitoring"):
			dpg.add_text('Testing 2')
	
dpg.set_item_pos('tab_bar', [100, 100]) # <- Not working

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()