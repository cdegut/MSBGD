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
    
    with dpg.theme(tag = "light"):
        with dpg.theme_component(0):
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvThemeCol_Text                   , (0.00 * 255, 0.00 * 255, 0.00 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled           , (0.60 * 255, 0.60 * 255, 0.60 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg               , (0.94 * 255, 0.94 * 255, 0.94 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg                , (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg                , (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.98 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_Border                 , (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.30 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow           , (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg                , (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered         , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.40 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive          , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.67 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg                , (0.96 * 255, 0.96 * 255, 0.96 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive          , (0.82 * 255, 0.82 * 255, 0.82 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed       , (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.51 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg              , (0.86 * 255, 0.86 * 255, 0.86 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg            , (0.98 * 255, 0.98 * 255, 0.98 * 255, 0.53 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab          , (0.69 * 255, 0.69 * 255, 0.69 * 255, 0.80 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered   , (0.49 * 255, 0.49 * 255, 0.49 * 255, 0.80 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive    , (0.49 * 255, 0.49 * 255, 0.49 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark              , (0.26 * 255, 0.59 * 255, 0.98 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab             , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.78 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive       , (0.46 * 255, 0.54 * 255, 0.80 * 255, 0.60 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_Button                 , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.40 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered          , (0.26 * 255, 0.59 * 255, 0.98 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive           , (0.06 * 255, 0.53 * 255, 0.98 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_Header                 , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.31 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered          , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.80 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive           , (0.26 * 255, 0.59 * 255, 0.98 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_Separator              , (0.39 * 255, 0.39 * 255, 0.39 * 255, 0.62 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered       , (0.14 * 255, 0.44 * 255, 0.80 * 255, 0.78 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive        , (0.14 * 255, 0.44 * 255, 0.80 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip             , (0.35 * 255, 0.35 * 255, 0.35 * 255, 0.17 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered      , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.67 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive       , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.95 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_Tab                    , (0.76 * 255, 0.80 * 255, 0.84 * 255, 0.93 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered             , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.80 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive              , (0.60 * 255, 0.73 * 255, 0.88 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocused           , (0.92 * 255, 0.93 * 255, 0.94 * 255, 0.99 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabUnfocusedActive     , (0.74 * 255, 0.82 * 255, 0.91 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_DockingPreview         , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.22 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_DockingEmptyBg         , (0.20 * 255, 0.20 * 255, 0.20 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_PlotLines              , (0.39 * 255, 0.39 * 255, 0.39 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_PlotLinesHovered       , (1.00 * 255, 0.43 * 255, 0.35 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram          , (0.90 * 255, 0.70 * 255, 0.00 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogramHovered   , (1.00 * 255, 0.45 * 255, 0.00 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TableHeaderBg          , (0.78 * 255, 0.87 * 255, 0.98 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderStrong      , (0.57 * 255, 0.57 * 255, 0.64 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TableBorderLight       , (0.68 * 255, 0.68 * 255, 0.74 * 255, 1.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBg             , (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.00 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TableRowBgAlt          , (0.30 * 255, 0.30 * 255, 0.30 * 255, 0.09 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_TextSelectedBg         , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.35 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_DragDropTarget         , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.95 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_NavHighlight           , (0.26 * 255, 0.59 * 255, 0.98 * 255, 0.80 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_NavWindowingHighlight  , (0.70 * 255, 0.70 * 255, 0.70 * 255, 0.70 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_NavWindowingDimBg      , (0.20 * 255, 0.20 * 255, 0.20 * 255, 0.20 * 255))
            dpg.add_theme_color(dpg.mvThemeCol_ModalWindowDimBg       , (0.20 * 255, 0.20 * 255, 0.20 * 255, 0.35 * 255))
            dpg.add_theme_color(dpg.mvPlotCol_FrameBg       , (1.00 * 255, 1.00 * 255, 1.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBg        , (255,255,255,255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_PlotBorder    , (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.00 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBg      , (1.00 * 255, 1.00 * 255, 1.00 * 255, 0.98 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendBorder  , (0.82 * 255, 0.82 * 255, 0.82 * 255, 0.80 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_LegendText    , (0.00 * 255, 0.00 * 255, 0.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_TitleText     , (0.00 * 255, 0.00 * 255, 0.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_InlayText     , (0.00 * 255, 0.00 * 255, 0.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisBg        , (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.00 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisBgActive  , (0.06 * 255, 0.53 * 255, 0.98 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisBgHovered , (0.26 * 255, 0.59 * 255, 0.98 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisGrid      , (0.00 * 255, 0.00 * 255, 0.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_AxisText      , (0.00 * 255, 0.00 * 255, 0.00 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Selection     , (0.82 * 255, 0.64 * 255, 0.03 * 255, 1.00 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Crosshairs    , (0.00 * 255, 0.00 * 255, 0.00 * 255, 0.50 * 255), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (240, 240, 240, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered, (240, 240, 240, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, (240, 240, 240, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeOutline, (100, 100, 100, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar, (248, 248, 248, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered, (209, 209, 209, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected, (209, 209, 209, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_Link, (66, 150, 250, 100), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkHovered, (66, 150, 250, 242), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkSelected, (66, 150, 250, 242), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_Pin, (66, 150, 250, 160), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_PinHovered, (66, 150, 250, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_BoxSelector, (90, 170, 250, 30), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_BoxSelectorOutline, (90, 170, 250, 150), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_GridBackground, (225, 225, 225, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_GridLine, (180, 180, 180, 100), category=dpg.mvThemeCat_Nodes)
