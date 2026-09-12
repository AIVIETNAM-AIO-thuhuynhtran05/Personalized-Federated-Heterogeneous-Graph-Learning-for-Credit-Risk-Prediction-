import numpy as np
import pandas as pd
from src.preprocessing.clean_application import drop_high_missing_columns
from src.preprocessing.feature_transformer import engineer_application_features

def test_drop_only_columns_above_80_percent_missing():
    data = pd.DataFrame({"SK_ID_CURR": range(5), "TARGET": [0, 0, 1, 0, 1],
                         "at_threshold": [1, np.nan, np.nan, np.nan, np.nan],
                         "above_threshold": [np.nan] * 5})
    cleaned, dropped = drop_high_missing_columns(data, threshold=0.80)
    assert "at_threshold" in cleaned
    assert dropped == ["above_threshold"]

def test_feature_engineering_preserves_partition_columns():
    data = pd.DataFrame({"AMT_CREDIT": [100.0], "AMT_INCOME_TOTAL": [50.0],
                         "REGION_RATING_CLIENT_W_CITY": [2],
                         "OCCUPATION_TYPE": ["Drivers"]})
    result = engineer_application_features(data)
    assert result.loc[0, "CREDIT_INCOME_RATIO"] == 2.0
    assert result.loc[0, "OCCUPATION_TYPE"] == "Drivers"
