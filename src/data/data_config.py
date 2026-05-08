datasets_csv_dict = {
    "Australia Bundoora": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTl3ZMKUEqYeXJe1b8A4IbfYIKjWlm0lR61glDoXOEfHxsmDUv1ZZ2IK2GpjkH2fZ6fvX3NaCOryqzW/pub?gid=751874949&single=true&output=csv",
    "Australia Albury-Wodonga": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQmRJCXCBp-qsY4LQrf8x_zJax_5FAnZDl6-sv1zje9m0pCM7hore-cjS3zlzJezgHIm6h81KY1hsEz/pub?gid=1184660391&single=true&output=csv",
    "Australia Bendigo": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQgXCJsm0V7ylsqvzRzK_LHZzky0lABeXvRuiqRWzDumN1Y8i8xul-Ih1ERIU1v-C46AKISnOOzBmtb/pub?gid=1902219272&single=true&output=csv",
    "morocco_zone_1": "https://docs.google.com/spreadsheets/d/e/2PACX-1vSgwB47qVFZcr1Aq--UWxZ6fDi9CGLZm-1i8QoMgfdaHUbV8EqSli3ayPxYYxD8kqfYYHD41uuNxbjZ/pub?gid=1952392108&single=true&output=csv",
    "Morocco Zone 2": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQT1DfqAB5Yec8MIQ_E5A8w-SXNcRmTwbXsv2W-ZT1ZcXN_G83BHlb6QBgnWkO-MpH3oVgfLoE0SnLx/pub?gid=1952392108&single=true&output=csv",
    "Morocco Zone 3": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSHw5k7n3_RM6ksGbvdQJsa1i9-zF-18CFLCFnXFkCxQwqLcQ4Wu2_8EF2H1lF02ih2NLL9BDecFzQ/pub?gid=1952392108&single=true&output=csv",
    "italy_temp": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRFF6SXvGbQgQG1bh0hwbXgVWpUU_UG8OQAVhHcNAcAFT5x-XoIYxMAeF-goym6_wNhJEwQd3iGHp9b/pub?gid=791211028&single=true&output=csv",

    # "load_consumption_real": "load_consumption_real"

    # 'DEKENERG_ZONE2_S_PEVRAOBL': '/Users/nikitasavvin/Desktop/Business/repo/collection_rus_energy_data/result_end/DEKENERG_ZONE2_S_PEVRAOBL.csv',
    # 'ARHENERG_ZONE1_E_PARHENER': '/Users/nikitasavvin/Desktop/Business/repo/collection_rus_energy_data/result_end/ARHENERG_ZONE1_E_PARHENER.csv',
    # 'DEKENERG_ZONE2_S_PAMURENE': '/Users/nikitasavvin/Desktop/Business/repo/collection_rus_energy_data/result_end/DEKENERG_ZONE2_S_PAMURENE.csv',
    "russia_elista": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTK8aQxQ15u_BZDSe1Qz5HW1gm-C7pLCEoJ8OrBskXMFgOzgUblKdJzGMUNFCnHe8JjPOBJzMcp1W8V/pub?gid=403462504&single=true&output=csv",


}

datasets_col_mapping = {
    "morocco_zone_1": {
        "time": "Datetime",
        "consumption": "consumption",
        "temperature": "Temperature"

    },
    "italy": {
        "time": "time",
        "consumption": "load_consumption",
        "temperature": "temperature"

    },
    "russia_elista": {
        "time": "datetime",
        "consumption": "value",
        "temperature": "temp"

    },
}
