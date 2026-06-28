from pathlib import Path
import re


VARIABLE_NAME_MAP = {
    "GoodStudent": "is_good_student",
    "Age": "driver_age_group",
    "SocioEcon": "socioeconomic_status",
    "RiskAversion": "risk_aversion_profile",
    "VehicleYear": "vehicle_age_category",
    "ThisCarDam": "current_car_damage",
    "RuggedAuto": "vehicle_ruggedness",
    "Accident": "accident_severity",
    "MakeModel": "vehicle_class",
    "DrivQuality": "driving_quality",
    "Mileage": "annual_mileage",
    "Antilock": "has_antilock_brakes",
    "DrivingSkill": "driving_skill_level",
    "SeniorTrain": "has_senior_driver_training",
    "ThisCarCost": "current_car_repair_cost",
    "Theft": "vehicle_theft_occurred",
    "CarValue": "vehicle_market_value",
    "HomeBase": "residence_environment",
    "AntiTheft": "has_anti_theft_system",
    "PropCost": "property_damage_cost",
    "OtherCarCost": "other_vehicle_damage_cost",
    "OtherCar": "other_vehicle_involved",
    "MedCost": "medical_cost",
    "Cushioning": "vehicle_safety_cushioning",
    "Airbag": "has_airbag",
    "ILiCost": "injury_liability_cost",
    "DrivHist": "driving_history_incidents",
}


def rename_bif_variables(
    input_path: Path,
    output_path: Path,
    name_map: dict[str, str],
) -> None:
    text = input_path.read_text(encoding="utf-8")

    # Replace only whole identifiers (avoid partial matches)
    for old, new in sorted(name_map.items(), key=lambda x: -len(x[0])):
        pattern = rf"\b{re.escape(old)}\b"
        text = re.sub(pattern, new, text)

    output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    dataset_name = "insurance.bif"

    path = Path(__file__).resolve().parents[2] / "datasets" / dataset_name
    
    input_bif = Path(path)
    output_bif = Path(path)

    rename_bif_variables(
        input_path=input_bif,
        output_path=output_bif,
        name_map=VARIABLE_NAME_MAP,
    )

    print(f"Renamed BIF written to: {output_bif}")