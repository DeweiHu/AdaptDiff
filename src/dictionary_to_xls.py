import openpyxl
import pickle
from scipy import stats


path = "/media/dewei/New Volume/MetaSyn/baseline/seg_maps/"
eval_type = "hausdorff"

def list_mean(number_list):
    mean = round(sum(number_list) / len(number_list), 4)
    return mean

def list_p_value(baseline_list, target_list):
    t_stat, p_value = stats.ttest_ind(target_list, baseline_list)
    return p_value


workbook = openpyxl.Workbook()
worksheet = workbook.active

# create a table 
worksheet["A1"] = "Methods"
worksheet["B1"] = "aria_amd"
worksheet["C1"] = "aria_diabetic"
worksheet["D1"] = "hrf_control"
worksheet["E1"] = "hrf_diabetic"
worksheet["F1"] = "hrf_glaucoma"
worksheet["G1"] = "octa500"
worksheet["H1"] = "rose"

worksheet["A2"] = "baseline"
worksheet["A3"] = "reg"
worksheet["A4"] = "bigaug"
worksheet["A5"] = "masf"
worksheet["A6"] = "vft"
worksheet["A7"] = "oracle"


# iterate 
start_column = "B"
end_column = "H"

start_ascii = ord(start_column)
end_ascii = ord(end_column)

for row in range(2, 7 + 1):
    method = worksheet[f"A{row}"].value
    
    for col_ascii in range(start_ascii, end_ascii + 1):
        col = chr(col_ascii)
        dataset = worksheet[f"{col}1"].value

        position = f"{col}{row}"

        target_file = f"{method}_{dataset}.pickle"
        with open(path + target_file, "rb") as handle:
            target_dict = pickle.load(handle)
        
        mean = list_mean(target_dict[eval_type])
        
        if method == "vft":

            baseline_file = f"baseline_{dataset}.pickle"
            with open(path + baseline_file, "rb") as handle:
                baseline_dict = pickle.load(handle)

            p_value = list_p_value(baseline_dict[eval_type], target_dict[eval_type])
            
            if p_value < 0.05:
                mark = "*"
            else:
                mark = ""

            worksheet[position] = f"{mean}{mark}"
        
        else:
            worksheet[position] = f"{mean}"



workbook.save(path + f"{eval_type}_source=fp.xlsx")

