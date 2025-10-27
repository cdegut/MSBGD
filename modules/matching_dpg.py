import dearpygui.dearpygui as dpg
from modules.rendercallback import RenderCallback
from modules.matching import draw_mz_lines, redraw_blocks, update_peak_starting_points, draw_mbg, print_to_terminal, refine_matching, show_projection
from modules.var import colors_list

def matching_window(render_callback:RenderCallback):
    spectrum = render_callback.spectrum
    with dpg.child_window(label="Peak matching", tag="Peak matching"):
        # Create a plot for the raw data
        with dpg.plot(label="Peak matching", width=1430, height=600, tag="peak_matching_plot") as plot2:
            # Add x and y axes
            dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag="x_axis_plot3")
            dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis_plot3")
            dpg.add_line_series([], [], label="Corrected Data Series", parent="y_axis_plot3", tag="corrected_series_plot3")
            dpg.add_line_series([], [], label="MBG", parent="y_axis_plot3", tag="MBG_plot3", show=False)
        with dpg.group(horizontal=True, horizontal_spacing= 25):
            with dpg.group(horizontal=False):
                dpg.add_button(label="Show fitted peaks", callback=update_peak_starting_points, user_data=render_callback)
                dpg.add_text("Tolerance:")
                dpg.add_input_int(label="m/z", default_value=25,min_value=1 , max_value=200, tag="block_width", width=100)
                dpg.add_text("Refine within:")
                dpg.add_input_int(label="Da", default_value=200, min_value=1 , max_value=1000, tag="refinement_width", width=100)
                dpg.add_checkbox(label="Show base projection", default_value=False, tag="show_projection", callback=show_projection, user_data=render_callback)
                # dpg.add_checkbox(label="Show Centers instead", default_value=False, tag="show_centers", callback=update_peak_starting_points, user_data=render_callback)
                # dpg.add_input_int(label="Width", default_value=1, min_value=1 , max_value=100, tag="center_width", width=100)
                dpg.add_checkbox(label="Hide High error", default_value=False, tag="hide_high_error", callback=update_peak_starting_points, user_data=render_callback)
                dpg.add_button(label="Draw MBG", callback=draw_mbg, user_data=render_callback)
                dpg.add_button(label="Print to term", callback=print_to_terminal, user_data=render_callback)
                
            for i in range(5):
                with dpg.child_window(height=200, width=220, tag = f"theorical_peaks_window_{i}"):
                    with dpg.theme(tag=f"theme_peak_window_{i}"):
                        with dpg.theme_component():
                            dpg.add_theme_color(dpg.mvThemeCol_Border, colors_list[i], category=dpg.mvThemeCat_Core)
                    
                    dpg.add_text(f"Peak Set {i}", tag=f"rmsd_{i}")
                    
                    with dpg.group(horizontal=True):
                        dpg.add_input_int(label="MW", default_value=549000, tag=f"molecular_weight_{i}", step = 100, width = 125, callback=draw_mz_lines, user_data=(render_callback, i))
                        dpg.add_button(label="Refine", callback=refine_matching, user_data=(render_callback, i))
                        
                    
                    dpg.add_input_int(label="Charges", default_value=52, tag=f"charges_{i}", width = 125, callback=draw_mz_lines,  user_data=(render_callback, i))
                    dpg.add_input_int(label="# Peaks", default_value=5, tag=f"nb_peak_show_{i}",step = 1, width = 125, callback=draw_mz_lines, user_data=(render_callback, i))
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("Compare with set:")
                        dpg.add_input_int(default_value=0, tag=f"compare_set_{i}", width = 80, callback=draw_mz_lines,  user_data=(render_callback, i))
                    dpg.add_text("", tag = f"MW_diff_{i}")
                    dpg.add_text("GeoMean of integral ratios:")
                    dpg.add_text("", tag = f"Integral_ratio_{i}")
                    dpg.add_table(header_row=False, row_background=True, tag=f"theorical_peak_table_{i}")
                    dpg.bind_item_theme(f"theorical_peaks_window_{i}", f"theme_peak_window_{i}")
    
    dpg.bind_item_theme("corrected_series_plot3", "data_theme")
    dpg.bind_item_theme("MBG_plot3", "matching_MBG_theme")