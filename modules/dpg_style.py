import dearpygui.dearpygui as dpg


def create_styles():
    with dpg.theme(tag="general_theme"):
        with dpg.theme_component(dpg.mvAll):
            #dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (131, 184, 198), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)

    with dpg.theme(tag="data_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (100, 100, 100), category=dpg.mvThemeCat_Plots)

    with dpg.theme(tag = "filtered_data_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (250,120,14), category=dpg.mvThemeCat_Plots)

    with dpg.theme(tag = "baseline_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (195, 242, 197), category=dpg.mvThemeCat_Plots)
    
    with dpg.theme(tag = "matching_MBG_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (195, 242, 197), category=dpg.mvThemeCat_Plots)

    with dpg.theme(tag = "fitting_MBG_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (64, 242, 197), category=dpg.mvThemeCat_Plots)

    with dpg.theme(tag = "table_row_bg_grey"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg, (100, 100, 100), category=dpg.mvThemeCat_Core)
    
    with dpg.theme(tag = "residual_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (98, 109, 126), category=dpg.mvThemeCat_Plots)

    with dpg.theme(tag = "fitted_peaks_theme"):
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (240, 145, 61), category=dpg.mvThemeCat_Plots)
