import sys
import pandas as pd
sys.path.append("/home/ekasteleyn/aurora_thesis/thesis/steering/scripts/oscillation_calculator")
from evaluate_all_steered import evaluate_set

def main():
    print("Evaluating negative AO alphas in AO_1encoder(2)...")
    df = evaluate_set("AO", "AO", "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)")
    
    # Also evaluate cont1 and cont10 for indices just in case
    evaluate_set("AO", "AO", "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_cont")
    evaluate_set("AO", "AO", "/home/ekasteleyn/aurora_thesis/thesis/steering/vectors/AO_1encoder(2)_cont10")

if __name__ == "__main__":
    main()
