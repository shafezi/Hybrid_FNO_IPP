"""Optimal hyperparameter dictionaries for different criteria."""

ALL_OPTIMAL = {
    "rmse": {
        5: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"matern","nu":0.5,"ls":0.10},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"matern","nu":0.5,"ls":0.10},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"matern","nu":0.5,"ls":0.02},
        },
        10: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"matern","nu":0.5,"ls":0.10},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"matern","nu":1.5,"ls":0.20},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"matern","nu":0.5,"ls":0.02},
        },
        20: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"matern","nu":1.5,"ls":0.02},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"matern","nu":2.5,"ls":0.05},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.30},
        },
        40: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.05},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"matern","nu":1.5,"ls":0.30},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"matern","nu":2.5,"ls":0.20},
        },
    },
    "acc": {
        5: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"matern","nu":2.5,"ls":0.05},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"matern","nu":1.5,"ls":0.30},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"matern","nu":0.5,"ls":0.02},
        },
        10: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"matern","nu":0.5,"ls":0.10},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.10},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"matern","nu":2.5,"ls":0.05},
        },
        20: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.10},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.05},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.30},
        },
        40: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.05},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.30},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"matern","nu":2.5,"ls":0.20},
        },
    },
    "mstruct": {
        5: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"matern","nu":2.5,"ls":0.30},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"matern","nu":0.5,"ls":0.10},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"matern","nu":1.5,"ls":0.05},
        },
        10: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.20},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"matern","nu":1.5,"ls":0.20},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"matern","nu":2.5,"ls":0.05},
        },
        20: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"matern","nu":2.5,"ls":0.05},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"matern","nu":2.5,"ls":0.05},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.05},
        },
        40: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.50},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.30},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.30},
        },
    },
    "m2": {
        5: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"matern","nu":2.5,"ls":0.02},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"matern","nu":0.5,"ls":0.10},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"matern","nu":1.5,"ls":0.05},
        },
        10: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"matern","nu":0.5,"ls":0.02},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"matern","nu":1.5,"ls":0.20},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"matern","nu":2.5,"ls":0.05},
        },
        20: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"matern","nu":2.5,"ls":0.05},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"matern","nu":0.5,"ls":0.50},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.05},
        },
        40: {
            "FNO-only":     {"forecast_mode":"fno",         "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "FNO + GP":     {"forecast_mode":"fno",         "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.05},
            "Persist-only": {"forecast_mode":"persistence", "use_policy":False, "kernel":"matern","nu":1.5,"ls":0.1},
            "Persist + GP": {"forecast_mode":"persistence", "use_policy":True,  "kernel":"matern","nu":1.5,"ls":0.30},
            "GP-only":      {"forecast_mode":"none",        "use_policy":True,  "kernel":"rbf",   "nu":1.5,"ls":0.30},
        },
    },
}
