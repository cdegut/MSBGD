from typing import List, Tuple
import dearpygui.dearpygui as dpg
from modules.data_structures import MSData
import numpy as np
from modules.rendercallback import RenderCallback
from modules.helpers import bi_gaussian
from modules.var import colors_list
from sklearn.linear_model import LinearRegression


def draw_mz_lines(sender = None, app_data = None, user_data:Tuple[RenderCallback,int] = None):
    render_callback: RenderCallback = user_data[0]
    k = user_data[1]
    
    mw:int = dpg.get_value(f"molecular_weight_{k}")
    charges:int = dpg.get_value(f"charges_{k}")
    nb_peak_show:int = dpg.get_value(f"nb_peak_show_{k}")
    render_callback.spectrum.matching_data[k] = [mw, charges, nb_peak_show]

    mz_l = []
    z_l = []
    z_mz = []

    for i in range(nb_peak_show):
        z = charges - i
        mz = (mw + z*0.007 ) / z
        mz_l.append(mz)
        z_l.append(z)
        z_mz.append((z, mz))

    for alias in dpg.get_aliases():
        if alias.startswith(f"mz_lines_{k}"):
            dpg.delete_item(alias)

    i = 0
    for line in mz_l:
        center_slice = render_callback.spectrum.baseline_corrected[:,1][(render_callback.spectrum.baseline_corrected[:,0] > line - 0.5) & (render_callback.spectrum.baseline_corrected[:,0] < line + 0.5)]
        max_y = max(center_slice) if len(center_slice) > 0 else 0
        dpg.draw_line((line, -50), (line, max_y), parent="peak_matching_plot", tag=f"mz_lines_{k}_{z_l[i]}", color=colors_list[k], thickness=1)
        i += 1
   
    update_theorical_peak_table(k, mz_l, z_l)
    render_callback.mz_lines[k] = z_mz
    redraw_blocks(render_callback)

def update_theorical_peak_table(k:int, mz_list:List[float], z_list): 
    dpg.delete_item(f"theorical_peak_table_{k}", children_only=True)

    dpg.add_table_column(parent = f"theorical_peak_table_{k}")
    dpg.add_table_column(parent = f"theorical_peak_table_{k}")
    dpg.add_table_column(parent = f"theorical_peak_table_{k}")


    row = len(mz_list)//3 + (len(mz_list)%3 >0)
    for r in range(0, row ):
        with dpg.table_row(parent = f"theorical_peak_table_{k}", tag = f"theorical_peak_table_{k}_{r}"):
            dpg.bind_item_theme( f"theorical_peak_table_{k}_{r}", "table_row_bg_grey")
            for n in range(0, 3):
                try:
                    dpg.add_text(f"{z_list[r*3+n]}+")
                except IndexError:
                    pass          
        
        with dpg.table_row(parent = f"theorical_peak_table_{k}"):
            for n in range(0, 3):
                try:
                    dpg.add_text(f"{int(mz_list[r*3+n])}")
                except IndexError:
                    pass    

def update_peak_starting_points(sender = None, app_data= None, user_data:RenderCallback = None):
    spectrum = user_data.spectrum 
    regression_projection = True

    if dpg.get_value("show_centers"):
        center_width = dpg.get_value("center_width") /100
        for peak in spectrum.peaks:
            if not spectrum.peaks[peak].fitted:
                continue
            apex = spectrum.peaks[peak].x0_refined
            sigma_L = spectrum.peaks[peak].sigma_L
            sigma_R = spectrum.peaks[peak].sigma_R
            spectrum.peaks[peak].start_range = (apex - sigma_L* center_width, apex + sigma_R * center_width)

    elif regression_projection:
        for peak in spectrum.peaks:
            if not spectrum.peaks[peak].fitted:
                continue

            A= spectrum.peaks[peak].A_refined
            apex = spectrum.peaks[peak].x0_refined
            start = apex
            sigma_L = spectrum.peaks[peak].sigma_L

            while True:
                A_current = bi_gaussian(start, spectrum.peaks[peak].A_refined, apex, spectrum.peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R)
                if A_current <= 0.8 * A:
                    break
                start -= 0.02
            mz80pcs = start


            while True:
                A_current = bi_gaussian(start, spectrum.peaks[peak].A_refined, apex, spectrum.peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R)
                if A_current <= 0.20 * A:
                    break
                start -= 0.02
            mz25pcs = start

            sample_points = np.linspace(mz80pcs, mz25pcs, 10)
            mz_samples = []
            A_samples = []

            for sample_mz in sample_points:
                A_sample = bi_gaussian(sample_mz, spectrum.peaks[peak].A_refined, apex, spectrum.peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R)
                mz_samples.append(sample_mz)
                A_samples.append(A_sample)

            X = np.array(mz_samples).reshape(-1, 1)
            y = np.array(A_samples)
            reg = LinearRegression().fit(X, y)

            a = reg.coef_[0]
            b = reg.intercept_

            x0 = -b / a

            spectrum.peaks[peak].regression_fct = (a, b)

    else:

        for peak in spectrum.peaks:
            if not spectrum.peaks[peak].fitted:
                continue

            A= spectrum.peaks[peak].A_refined
            apex = spectrum.peaks[peak].x0_refined
            start = apex -  spectrum.peaks[peak].sigma_L
            
            while  bi_gaussian(start, spectrum.peaks[peak].A_refined, apex, spectrum.peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R) > 0.1 * A:       
                start -= 0.01
            start10pcs = start
            while  bi_gaussian(start, spectrum.peaks[peak].A_refined, apex, spectrum.peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R) > 5 * A:       
                start -= 0.01
            start1pcs = start
            spectrum.peaks[peak].start_range = (start1pcs, start10pcs)
        
    redraw_blocks(user_data)

def draw_mbg(sender = None, app_data = None, user_data:RenderCallback = None):

    
    for alias in dpg.get_aliases():
        if alias.startswith("MBG_plot3"):
            dpg.delete_item("MBG_plot3")
            return
        
    spectrum = user_data.spectrum
    x_data = spectrum.working_data[:,0]
    mbg = spectrum.calculate_mbg(x_data)
    dpg.show_item("MBG_plot3")
    dpg.set_value("MBG_plot3", [x_data.tolist(),mbg.tolist()])


def redraw_blocks(render_callback:RenderCallback):
    spectrum = render_callback.spectrum
    block_height = max(spectrum.baseline_corrected[:,1])  /20
    block_width = dpg.get_value("block_width")
    show_projection = dpg.get_value("show_projection")

    # Delete previous peaks and annotation
    for alias in dpg.get_aliases():
        if alias.startswith("fitted_peak_matching") or alias.startswith("peak_annotation_matching")  or alias.startswith("fitted_regression_"):
            dpg.delete_item(alias)   
    
    for peak in spectrum.peaks:
        if not spectrum.peaks[peak].fitted:
            continue
        spectrum.peaks[peak].matched_with = [0,0,0]  # Reset matched_with for all peaks

        regression_0 = - spectrum.peaks[peak].regression_fct[1] / spectrum.peaks[peak].regression_fct[0]
        regression_x = (regression_0 + spectrum.peaks[peak].sigma_L * 5)
        regression_y = (spectrum.peaks[peak].regression_fct[0] * regression_x + spectrum.peaks[peak].regression_fct[1])

        # Regression lines
        if show_projection:
            dpg.draw_line((regression_0, 0), (regression_x, regression_y), parent="peak_matching_plot", color=(75,25,25), thickness=1, tag=f"fitted_regression_{peak}")
        
        # non matched blocks       
        dpg.draw_line((regression_0, block_height), (regression_0, -25), parent="peak_matching_plot", color=(246, 32, 24,128), thickness=block_width, tag=f"fitted_peak_matching_{peak}")        
        dpg.add_plot_annotation(label=f"Peak {peak}", default_value=(regression_0, 0), offset=(15, 15), color=(100, 100, 100), clamped=False, parent="peak_matching_plot", tag=f"peak_annotation_matching_{peak}_gray")

        for k in range(0, len(render_callback.mz_lines)):
            mz_lines = render_callback.mz_lines[k]
            color = colors_list[k]
            transparent_color = list(color)
            transparent_color.append(100)
            transparent_color = tuple(transparent_color)

            # find matched blocks
            for z_mz in mz_lines:
                if z_mz[1] >  regression_0 - block_width/2 and z_mz[1] < regression_0 + block_width/2:
                    dpg.delete_item(f"fitted_peak_matching_{peak}")
                    dpg.delete_item(f"mz_lines_{k}_{z_mz[0]}")
                    dpg.draw_line((z_mz[1], -50), (z_mz[1], block_height*1.5), parent="peak_matching_plot", tag=f"mz_lines_{k}_{z_mz[0]}", color=colors_list[k], thickness=1)
                    dpg.draw_line((regression_0, block_height), (regression_0, -25), parent="peak_matching_plot", color=transparent_color, thickness=block_width, tag=f"fitted_peak_matching_{peak}")
                    spectrum.peaks[peak].matched_with = [k, z_mz[0], z_mz[1]*z_mz[0]]
                    #dpg.delete_item(f"peak_annotation_matching_{k}_{z_mz[0]}")
                    if not dpg.does_alias_exist(f"peak_annotation_matching_{k}_{z_mz[0]}"):
                        x0_fit = spectrum.peaks[peak].x0_refined
                        y_mbg = spectrum.calculate_mbg(x0_fit)
                        dpg.add_plot_annotation(label=f"{z_mz[0]}+", default_value=(x0_fit, y_mbg), offset=(15, -15), color=color, clamped=False, parent="peak_matching_plot", tag=f"peak_annotation_matching_{k}_{z_mz[0]}")

    # Draw the line annotations of unmatched peaks      
    for k in range(0, len(render_callback.mz_lines)):
        mz_lines = render_callback.mz_lines[k]
        color = colors_list[k]
        for z_mz in mz_lines:
            if not dpg.does_alias_exist(f"peak_annotation_matching_{k}_{z_mz[0]}"):           
                dpg.add_plot_annotation(label=f"{z_mz[0]}+", default_value=(z_mz[1], 0), offset=(-15, -15-k*15), color=colors_list[k], clamped=False, parent="peak_matching_plot", tag=f"peak_annotation_matching_{k}_{z_mz[1]}")
    
    matching_quality(render_callback)
    check_mass_difference(render_callback)

def show_projection(sender = None, app_data = None, user_data:RenderCallback = None):
    redraw_blocks(user_data)

def calculate_quality_score(render_callback:RenderCallback, k:int , shift = 0) -> float:
    spectrum = render_callback.spectrum
    mz_lines = render_callback.mz_lines[k]
    block_width = dpg.get_value("block_width")
    squares = []
    
    for z_mz in mz_lines:
        for peak in spectrum.peaks:
            if not spectrum.peaks[peak].fitted:
                continue

            regression_0 = (- spectrum.peaks[peak].regression_fct[1] / spectrum.peaks[peak].regression_fct[0] )
            z_mz_shifted = z_mz[1] + (shift / z_mz[0])
            if z_mz_shifted >  regression_0 - block_width/2 and z_mz_shifted < regression_0 + block_width/2:
                square_distance = (z_mz_shifted - regression_0)**2
                squares.append(square_distance)
    peaks = len(squares)
    if peaks> 0:
        rmsd = np.sqrt(np.mean(squares))
        score = (1 / rmsd) * 100 + (len(squares) * 20)
        
        return peaks, rmsd, score
    else:
        return 0, 0.0, 0.0
    
def matching_quality(render_callback:RenderCallback):
    for k in range(0, len(render_callback.mz_lines)):
        peaks, rmsd, score = calculate_quality_score(render_callback, k)
        
        if peaks > 1:
            dpg.set_value(f"rmsd_{k}", f"{peaks} - RMSD: {rmsd:.2f} Score: {score}")

        elif peaks == 1:
            dpg.set_value(f"rmsd_{k}", f"Only 1 peak match")
        else:
            dpg.set_value(f"rmsd_{k}", f"No matching peaks")

    
def refine_matching(sender = None, app_data = None, user_data:Tuple[RenderCallback,int] = None):
    refinement_width = dpg.get_value("refinement_width")
    k = user_data[1]
    local_score = []

    for shift in range(-refinement_width, refinement_width +1, 1):
        peaks, rmsd, score = calculate_quality_score( user_data[0], k, shift)
        local_score.append((shift, peaks, rmsd, score))

    max_peaks = max(local_score, key=lambda x: x[1])[1]
    best_rmsd = min([x for x in local_score if x[1] == max_peaks], key=lambda x: x[2])
    new_mw = dpg.get_value(f"molecular_weight_{k}") + best_rmsd[0]

    dpg.set_value(f"molecular_weight_{k}", new_mw)
    draw_mz_lines(sender = None, app_data = None, user_data=user_data)

def check_mass_difference(render_callback:RenderCallback):
    for k in range(0, 5):
        mw_set_i = dpg.get_value(f"compare_set_{k}")
        mw_set1 =dpg.get_value(f"molecular_weight_{mw_set_i}")
        mw = dpg.get_value(f"molecular_weight_{k}")
        diff = mw - mw_set1
        dpg.set_value(f"MW_diff_{k}", f"d{diff}")
    check_integral_ratio(render_callback)

def check_integral_ratio(render_callback:RenderCallback):
    for k in range(0, 5):
        compare_to = int(dpg.get_value(f"compare_set_{k}"))
        integral_set1 = []
        integral_set2 = []

        for peak in render_callback.spectrum.peaks:
            if not render_callback.spectrum.peaks[peak].fitted:
                continue
            matched = render_callback.spectrum.peaks[peak].matched_with
            if matched == [0,0,0]:
                continue

            if matched[0] == compare_to:
                integral_set1.append([matched[1], render_callback.spectrum.peaks[peak].integral])
            if matched[0] == k:
                integral_set2.append([matched[1], render_callback.spectrum.peaks[peak].integral])
        
        ratios = []
        zs = []
        for z1, int1 in integral_set1:
            for z2, int2 in integral_set2:
                if z1 == z2:
                    ratio = int1 / int2
                    ratios.append(ratio)
                    zs.append(z1)
                    
        if len(ratios) >0:
            geometric_mean = np.exp(np.mean(np.log(ratios)))
            dpg.set_value(f"Integral_ratio_{k}", f"{geometric_mean:.2f} using {len(zs)} peaks")

def print_to_terminal(sender = None, app_data = None, user_data:RenderCallback = None):
    spectrum = user_data.spectrum
    print("Matched Peaks:")
    if not spectrum.peaks:
        print("No peaks found.")
        return
    # Collect peaks grouped by the first value of matched_with
    grouped_peaks = {}
    for peak in spectrum.peaks:
        if not spectrum.peaks[peak].fitted:
            continue
        matched = spectrum.peaks[peak].matched_with
        if matched == [0,0,0]:
            continue
        group_key = matched[0]
        if group_key not in grouped_peaks:
            grouped_peaks[group_key] = []
        grouped_peaks[group_key].append({
            "peak": peak,
            "x0": spectrum.peaks[peak].x0_refined,
            "integral": spectrum.peaks[peak].integral,
            "matched_with": matched
        })

    for group, peaks in grouped_peaks.items():
        print(f"Mass group {group}: ")
        print("Peak:\tz:\tM:\tx0:\tIntegral:")
        for p in peaks:
            print(f"{p['peak']}\t{p['matched_with'][1]}\t{p['matched_with'][2]:.2f}\t{p['x0']:.2f}\t{p['integral']:.2f}")

    for group, peaks in grouped_peaks.items():
        print(f"Integrals for mass group {group}:")
        for p in peaks:
            print(f"{p['integral']:.2f}")

    
