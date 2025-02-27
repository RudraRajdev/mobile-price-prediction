import pandas as pd
import joblib
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Load model and scaler
model = joblib.load('mobile_price_model.pkl')
scaler = joblib.load('mobile_price_scaler.pkl')

# Feature information with descriptions and units
FEATURE_INFO = {
    'battery_power': {'desc': 'Battery Capacity', 'unit': 'mAh', 'range': (500, 4000)},
    'clock_speed': {'desc': 'Processor Speed', 'unit': 'GHz', 'range': (0.5, 3.5)},
    'fc': {'desc': 'Front Camera', 'unit': 'MP', 'range': (0, 32)},
    'int_memory': {'desc': 'Internal Storage', 'unit': 'GB', 'range': (2, 256)},
    'm_dep': {'desc': 'Device Thickness', 'unit': 'cm', 'range': (0.1, 2.0)},
    'mobile_wt': {'desc': 'Device Weight', 'unit': 'grams', 'range': (80, 300)},
    'n_cores': {'desc': 'Processor Cores', 'unit': 'cores', 'range': (1, 8)},
    'pc': {'desc': 'Primary Camera', 'unit': 'MP', 'range': (0, 108)},
    'px_height': {'desc': 'Screen Height', 'unit': 'pixels', 'range': (500, 3000)},
    'px_width': {'desc': 'Screen Width', 'unit': 'pixels', 'range': (500, 3000)},
    'ram': {'desc': 'Memory Size', 'unit': 'MB', 'range': (512, 8192)},
    'sc_h': {'desc': 'Screen Height', 'unit': 'cm', 'range': (5, 25)},
    'sc_w': {'desc': 'Screen Width', 'unit': 'cm', 'range': (3, 18)},
    'talk_time': {'desc': 'Battery Life', 'unit': 'hours', 'range': (2, 24)}
}

PRICE_CATEGORIES = {
    0: ("Low Cost", Fore.GREEN),
    1: ("Medium Cost", Fore.BLUE),
    2: ("High Cost", Fore.YELLOW),
    3: ("Very High Cost", Fore.RED)
}

def display_welcome():
    print(Fore.CYAN + "\n" + "="*50)
    print(Fore.CYAN + " Mobile Phone Price Predictor ".center(50))
    print(Fore.CYAN + "="*50)
    print("\nPlease enter the following specifications:")
    print(Fore.MAGENTA + "Note: Use numbers only, decimals are allowed where applicable\n")

def get_feature_input(feature):
    info = FEATURE_INFO[feature]
    while True:
        try:
            prompt = f"{info['desc']} ({info['unit']}) [{info['range'][0]}-{info['range'][1]}]: "
            value = float(input(Fore.WHITE + prompt))
            
            if not (info['range'][0] <= value <= info['range'][1]):
                print(Fore.RED + f"Please enter a value between {info['range'][0]} and {info['range'][1]}")
                continue
                
            return value
        except ValueError:
            print(Fore.RED + "Invalid input. Please enter a valid number.")

def collect_specs():
    return {feature: get_feature_input(feature) for feature in FEATURE_INFO}

def predict_price(specs):
    df = pd.DataFrame([specs])
    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)[0]
    return prediction

def display_result(prediction):
    category, color = PRICE_CATEGORIES[prediction]
    print(Fore.CYAN + "\n" + "-"*50)
    print(color + f" Predicted Price Category: {category} ".center(50))
    print(Fore.CYAN + "-"*50 + "\n")

def main():
    display_welcome()
    while True:
        specs = collect_specs()
        prediction = predict_price(specs)
        display_result(prediction)
        
        if input(Fore.WHITE + "Predict another phone? (y/n): ").lower() != 'y':
            print(Fore.GREEN + "\nThank you for using the Mobile Price Predictor!")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.RED + "\n\nOperation cancelled by user. Exiting...")
    except Exception as e:
        print(Fore.RED + f"\nAn error occurred: {str(e)}")