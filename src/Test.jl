using Revise
using DataFrames
using SerinGrid

# 1. Mocking some regional machine spare parts data
mock_data = DataFrame(
    ItemID = ["SP-001", "SP-002", "SP-003", "SP-004", "SP-005"],
    Revenue = [80000.0, 15000.0, 4000.0, 800.0, 200.0], # Matches 80/15/5 distribution curve
    DemandHistory = [
        [100, 102, 98, 101], # High revenue, completely stable (A-V1)
        [50, 10, 85, 30],    # High revenue, moderately volatile (A-V3)
        [10, 12, 11, 9],     # Medium revenue, stable (B-V1)
        [2, 0, 15, 1],       # Low revenue, highly erratic/sporadic (C-V4)
        [0, 0, 1, 0]         # Very low revenue, extreme edge case (C-V4)
    ]
)

# 2. Run with default thresholds (80% Revenue, CV Limits: 0.2, 0.5, 1.0)
println("--- Running Default Analysis ---")
processed_df = classify_inventory!(copy(mock_data))
println(processed_df[:, [:ItemID, :Revenue, :CV, :SPARE_Grid]])

# 3. Programming dynamic user-defined volatility overrides
println("\n--- Running Custom Volatility Shift ---")
custom_settings = SPAREConfig(
    cv_low = 0.05, 
    cv_med = 0.6, 
    cv_high = 0.8
)

custom_processed_df = classify_inventory!(copy(mock_data), config=custom_settings)
println(custom_processed_df[:, [:ItemID, :Revenue, :CV, :SPARE_Grid]])
println("\n--- 3x4 SPARE Grid Summary (Counts & Dynamic Labels) ---")
# Pass the custom dataframe AND the custom configuration
summary_matrix = generate_grid_summary(custom_processed_df, config=custom_settings)
println(summary_matrix)