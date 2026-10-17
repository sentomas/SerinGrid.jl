# Load the v0.2.0 module directly from the relative src folder
include(joinpath(@__DIR__, "..", "src", "SerinGrid.jl"))
using .SerinGrid
using DataFrames
using Statistics
using Dates

# 48-Month (4-Year) Industrial Spare Parts Benchmark Dataset
d_fast = fill(50.0, 48) .+ [iseven(i) ? 4.0 : -4.0 for i in 1:48]
d_mod  = [20.0 + 8.0 * sin(2π * i / 12) for i in 1:48]
d_erratic = [i % 3 == 0 ? 12.0 : 1.0 for i in 1:48]
d_intermittent = zeros(Float64, 48); d_intermittent[18] = 1.0; d_intermittent[42] = 1.0
d_burst = zeros(Float64, 48); d_burst[36] = 15.0

df_spares = DataFrame(
    ItemID = [
        "SKU-01 (Fast Filter)",
        "SKU-02 (Mod Valve)",
        "SKU-03 (Lumpy Seal)",
        "SKU-04 (2-in-4Yr Motor - Trap!)",
        "SKU-05 (Sporadic Burst)"
    ],
    Revenue = [450000.0, 250000.0, 150000.0, 120000.0, 30000.0],
    DemandHistory = [d_fast, d_mod, d_erratic, d_intermittent, d_burst]
)

SerinGrid.classify_inventory!(df_spares)
df_ml = SerinGrid.export_ml_features(df_spares)
df_summary = SerinGrid.generate_grid_summary(df_spares)

# Define the log file path inside the same 'test' folder
log_path = joinpath(@__DIR__, "opsearch_validation_results.log")

# Helper function to format output for both the log file and terminal
function write_report(io::IO)
    println(io, "====================================================================================")
    println(io, " SERINGRID.JL v0.2.0 - OPSEARCH JOURNAL VALIDATION LOG")
    println(io, " Generated: ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
    println(io, "====================================================================================\n")

    println(io, "--- TABLE 1: RAW CV vs. HYBRID BAYESIAN CV & AI/ML ROUTING (48 MONTHS) ---")
    show(io, select(df_spares, :ItemID, :ZeroRatio, :CV_Raw, :CV, :CV_Upper95, :SPARE_Grid, :BoundaryRisk, :RoutedMLModel); allcols=true, allrows=true)
    println(io, "\n\n")

    println(io, "--- TABLE 2: 3x4 DYNAMIC SPARE GRID SUMMARY ---")
    show(io, df_summary; allcols=true, allrows=true)
    println(io, "\n\n")

    println(io, "--- TABLE 3: AI/ML FEATURE MATRIX EXPORT (export_ml_features) ---")
    show(io, df_ml; allcols=true, allrows=true)
    println(io, "\n")
end

# 1. Save results to the log file
open(log_path, "w") do io
    write_report(io)
end

# 2. Also display the results in the terminal
write_report(stdout)
println("\n[INFO] Validation results successfully saved to: ", log_path)
