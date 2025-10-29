"""
Código maestro para ejecutar el flujo completo de ML en Fase2_team42.
Desde la lectura de datos hasta la exportación de resultados de ML.
"""
import os
import pandas as pd

def main():
    import os
    # 1. EDA y preprocesamiento
    from src.data.eda import load_and_describe_data, plot_target_distribution, save_top_correlations, plot_top_features, split_and_save_train_test
    # Definir nombres de columnas (ajustar según el diccionario real)
    column_names = [
        "CSubType", "NumHouse", "AvgSzHouse", "AvgAge", "CMainType",
        "DemCatholic", "DemProtestant", "DemOthReligion", "DemNoReligion",
        "DemMarried", "DemCohabitation", "DemOtherRelation", "DemSingles",
        "DemNoChild", "DemWithChild", "DemHiLeEdu", "DemMiLeEdu", "DemLoLeEdu",
        "DemHiStatus", "DemEntrepreneur", "DemFarmer", "DemMidManager",
        "DemSkilledLab", "DemUnskilledLab", "DemSocClassA", "DemSocClassB1",
        "DemSocClassB2", "DemSocClassC", "DemSocClassD", "DemRenter",
        "DemHomeOwner", "Dem1Car", "Dem2Car", "Dem0Car", "DemPubHthInsur",
        "DemPrivHthInsur", "DemLowestInc", "DemLowInc", "DemMidInc",
        "DemHighInc", "DemHighestInc", "DemAvgIncome", "DemPurchPowerC",
        "ContrPrivIns", "ContrPrivInsFirm", "ContrPrivInsAgr", "ContrCarPol",
        "ContrDeVanPol", "ContrMotPol", "ContrLorPol", "ContrTrailPol",
        "ContrTractPol", "ContrAgrMacPol", "ContrMopPol", "ContrLifePol",
        "ContrPrivAccPol", "ContrFamAccPol", "ContrDisPol", "ContrFirePol",
        "ContrSurfPol", "ContrBoatPol", "ContrBicPol", "ContrProPol",
        "ContrSSPol", "NumPrivIns", "NumPrivInsFirm", "NumPrivInsAgr",
        "NumCarPol", "NumDeVanPol", "NumMotPol", "NumLorPol", "NumTrailPol",
        "NumTractPol", "NumAgrMacPol", "NumMopPol", "NumLifePol",
        "NumPrivAccPol", "NumFamAccPol", "NumDisPol", "NumFirePol",
        "NumSurfPol", "NumBoatPol", "NumBicPol", "NumProPol", "NumSSPol",
        "MoHoPol"
    ]
    input_path = os.path.join(os.path.dirname(__file__), 'data', 'insurance_company_original.csv')
    if not os.path.exists(input_path):
        input_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'insurance_company_original.csv')
    df = load_and_describe_data(input_path, column_names)
    plot_target_distribution(df, 'MoHoPol')
    top10 = save_top_correlations(df, 'MoHoPol', top_n=10)
    plot_top_features(df, top10.index, 'MoHoPol')
    train_df, test_df = split_and_save_train_test(df, 'MoHoPol', random_state=42)
    print("EDA y preprocesamiento completados.")

    # 2. Modelado y evaluación
    from src.models.ml_pipeline import load_train_test, get_features, train_and_evaluate
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'ml_data')
    df_train, df_test = load_train_test(data_dir, 'MoHoPol')
    df_train_fx, df_test_fx, FEATURES = get_features(df_train, df_test)
    X_train = df_train_fx[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_train = df_train_fx['MoHoPol'].astype(int)
    X_test = df_test_fx[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0)
    y_test = df_test_fx['MoHoPol'].astype(int)
    results = train_and_evaluate(X_train, y_train, X_test, y_test)
    print("Modelado y evaluación completados. Resultados exportados en reports.")

if __name__ == "__main__":
    main()
