def csv_info_filter(d: dict) -> dict:
    keys = d['column_info'].keys()
    for key in keys:
        d['column_info'][key].pop('required', None)
        d['column_info'][key].pop('nan_processing', None)
        d['column_info'][key].pop('encode_mapping', None)
    return d


def predict_info_filter(preprocessing_config: dict, cv_res: dict) -> dict:
    attr_from_old_dict = {'project_name', 'file_name'}
    pop_from_column_info = {'total', 'nan', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'encode_mapping', 'mode'}

    new_d = {attr: preprocessing_config[attr] for attr in attr_from_old_dict}.copy()
    new_d['column_info'] = {}
    keys = cv_res['feature']

    for key in keys:
        old_d_key = set(preprocessing_config['column_info'][key].keys())
        new_d['column_info'][key] = {
            'column_class': preprocessing_config['column_info'][key]['column_class'],
            'nan_processing': preprocessing_config['column_info'][key]['nan_processing'],
            'required': preprocessing_config['column_info'][key]['required'],
        }

        if new_d['column_info'][key]['column_class'] == "discrete variable":
            new_d_key = set(new_d['column_info'][key].keys())
            new_d['column_info'][key]['options'] = list(old_d_key - new_d_key - pop_from_column_info)

    new_d['columns'] = list(new_d['column_info'].keys())
    new_d['column_num'] = len(new_d['columns'])

    return new_d
