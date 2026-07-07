#==============================================================================#
# Module: SerinGrid.jl
# Author: Serin Thomas
# Date: 2026-Jul-07
#
# Description: 
# This module implements the SPARE Grid classification system for inventory 
# management. It categorizes regional machine spare parts based on demand 
# volatility (CV) and revenue contribution (ABC), providing a 12-grid matrix 
# designed to feed into predictive forecasting architectures.
#
# License: MIT License
# Copyright (c) 2026 Serin Thomas. All rights reserved.
# See the LICENSE file in the project root for full license information.
#==============================================================================#
module SerinGrid

using DataFrames
using Statistics

export SPAREConfig, classify_inventory!, calculate_cv, generate_grid_summary

"""
    SPAREConfig
"""
Base.@kwdef struct SPAREConfig
    # CV Volatility thresholds
    cv_low::Float64  = 0.20
    cv_med::Float64  = 0.50
    cv_high::Float64 = 1.00
    
    # Revenue splits
    abc_a::Float64   = 0.80
    abc_b::Float64   = 0.95
end

"""
    calculate_cv(demands)
"""
function calculate_cv(demands::AbstractVector{<:Real})::Float64
    avg = mean(demands)
    if avg <= 0.0
        return 0.0 
    end
    return std(demands) / avg
end

"""
    classify_inventory!(df::DataFrame; config::SPAREConfig)
"""
function classify_inventory!(df::DataFrame; config::SPAREConfig = SPAREConfig())::DataFrame
    # Safety checks
    required_cols = [:ItemID, :Revenue, :DemandHistory]
    for col in required_cols
        if !hasproperty(df, col)
            throw(ArgumentError("Input DataFrame is missing required column: $col"))
        end
    end

    # 1. Volatility Calculation (CV)
    df.CV = map(calculate_cv, df.DemandHistory)
    
    df.Volatility_Class = map(df.CV) do cv
        if cv <= config.cv_low
            return "V1"
        elseif cv <= config.cv_med
            return "V2"
        elseif cv <= config.cv_high
            return "V3"
        else
            return "V4"
        end
    end

    # 2. Revenue ABC Sorting & Calculation
    sort!(df, :Revenue, rev=true)
    total_rev = sum(df.Revenue)
    
    if total_rev <= 0.0
        throw(ArgumentError("Total revenue across all parts must be greater than zero."))
    end

    df.CumRevPct = cumsum(df.Revenue) ./ total_rev

    # 3. ABC Assignment
    df.ABC_Class = map(df.CumRevPct) do pct
        if pct <= config.abc_a
            return "A"
        elseif pct <= config.abc_b
            return "B"
        else
            return "C"
        end
    end

    # 4. Synthesize into the 12-Grid System
    df.SPARE_Grid = df.ABC_Class .* "-" .* df.Volatility_Class

    select!(df, Not(:CumRevPct))

    return df
end

"""
    generate_grid_summary(df::DataFrame; config::SPAREConfig)
"""
function generate_grid_summary(df::DataFrame; config::SPAREConfig = SPAREConfig())::DataFrame
    # 1. Create a template of all 12 combinations
    template = DataFrame(
        ABC_Class = repeat(["A", "B", "C"], inner=4),
        Volatility_Class = repeat(["V1", "V2", "V3", "V4"], outer=3)
    )
    
    # 2. Count the occurrences in the actual processed data
    counts = combine(groupby(df, [:ABC_Class, :Volatility_Class]), nrow => :Count)
    
    # 3. Join the counts to the template and fill missing values with 0
    merged = leftjoin(template, counts, on=[:ABC_Class, :Volatility_Class])
    merged.Count = coalesce.(merged.Count, 0)
    
    # 4. Pivot the table into the final 3x4 grid
    grid_3x4 = unstack(merged, :ABC_Class, :Volatility_Class, :Count)
    
    # 5. Force the correct column order
    select!(grid_3x4, :ABC_Class, :V1, :V2, :V3, :V4)
    
    # 6. Dynamically rename the columns using the active config thresholds
    rename!(grid_3x4, 
        :V1 => "V1 (CV <= $(config.cv_low))",
        :V2 => "V2 (CV <= $(config.cv_med))",
        :V3 => "V3 (CV <= $(config.cv_high))",
        :V4 => "V4 (CV > $(config.cv_high))"
    )
    
    return grid_3x4
end

end # module