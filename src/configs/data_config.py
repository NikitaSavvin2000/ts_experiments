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

    "russia_elista": "https://docs.google.com/spreadsheets/d/e/2PACX-1vTK8aQxQ15u_BZDSe1Qz5HW1gm-C7pLCEoJ8OrBskXMFgOzgUblKdJzGMUNFCnHe8JjPOBJzMcp1W8V/pub?gid=442094039&single=true&output=csv",
    "Istanbul_Traffic_Index": "https://docs.google.com/spreadsheets/d/e/2PACX-1vT6LSgQb7b1u-c_zBRhvkEA_CmNd6Cby2tlMLBwoX9GlSsUHjJvdh9Xz9zyQx5HbxQAUH2GEVrn95U7/pub?gid=2111659113&single=true&output=csv",
    "Air_Quality_India": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQTTyEI6cJP1q0SIUlXa46vpBFGdCe6BuI4Y9RxvJ8ZCkSarq85wFAlox6qbcVvsZ3fdHdErmugSJ2l/pub?gid=1652276659&single=true&output=csv",
    "Daily_Climate": "https://docs.google.com/spreadsheets/d/e/2PACX-1vT6-YtMLu5kDfAfr4ISYnMOulJf4IU3RNAe6vRo7adY_A4Hot6K3vWtO0HKsrAZjNng0cMQkxIxvQbp/pub?gid=81811996&single=true&output=csv",
    "NYC_Taxi_Traffic":"https://docs.google.com/spreadsheets/d/e/2PACX-1vQgq9c-tCaL3ZFDLoFvCaB-dvW1SPwO2u8RHYMGpNAzg4O0Bl-kjRaKatD2i-V2-uafFcNOKYcMytnZ/pub?gid=797724504&single=true&output=csv", # https://www.kaggle.com/datasets/julienjta/nyc-taxi-traffic
    # "NYC_Taxi_Traffic": "", # https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset
   "Weather": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRbZQWfPBOCnG6OQI5uzi886u0ZjOYeNdCcyJCx3up2bjjPdm7BX20akl8oSM2rjpiEoAtQmREK2s_7/pub?gid=2018623599&single=true&output=csv", # https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting
}

# Климат
# Энергетика
# Транспорт и логистика
# Медицина
# Продажи

# Climate
# Energy
# Transport and Logistics
# Healthcare
# Sales

datasets_col_mapping = {
    "morocco_zone_1": {
        "time": "Datetime",
        "target": "consumption",
        "temperature": "Temperature"

    },
    "italy": {
        "time": "time",
        "target": "load_consumption",
        "temperature": "temperature"

    },
    "russia_elista": {
        "time": "datetime",
        "target": "value",
        "temperature": "temp"

    },
    "Istanbul_Traffic_Index":{
        "time": "datetime",
        "target": "average_traffic_index",
        "temperature": "temp"

    },
}
