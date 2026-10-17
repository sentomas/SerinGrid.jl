module SerinGrid

using Statistics
using DataFrames

export SPAREConfig, calculate_cv, calculate_bayesian_bounds, 
       recommend_ml_model, classify_inventory!, generate_grid_summary,
       export_ml_features

Base.@kwdef struct SPAREConfig
    cv_low::Float64  = 0.20
    cv_med::Float64  = 0.50
    cv_high::Float64 = 1.00
    abc_a::Float64   = 0.80
    abc_b::Float64   = 0.95
    use_hybrid_bayesian::Bool = true
    intermittent_threshold::Float64 = 1.0
    prior_alpha::Float64 = 2.0
    prior_beta::Float64  = 2.0
end

function calculate_cv(demands::AbstractVector{<:Real}; config::SPAREConfig = SPAREConfig())::Float64
    n = length(demands)
    n == 0 && return 0.0
    avg = mean(demands)
    avg <= 0.0 && return 0.0

    if config.use_hybrid_bayesian && avg < config.intermittent_threshold
        alpha_post = config.prior_alpha + sum(demands)
        return 1.0 / sqrt(alpha_post)
    end
    return std(demands) / avg
end

function calculate_bayesian_bounds(demands::AbstractVector{<:Real}; config::SPAREConfig = SPAREConfig())
    n = length(demands)
    sum_d = sum(demands)
    if sum_d <= 0.0 || n == 0
        return (cv_post = 0.0, cv_lower = 0.0, cv_upper = 0.0)
    end
    alpha_post = config.prior_alpha + sum_d
    std_alpha  = sqrt(alpha_post)
    alpha_low  = max(0.5, alpha_post - 1.96 * std_alpha)
    alpha_high = alpha_post + 1.96 * std_alpha

    return (
        cv_post  = 1.0 / sqrt(alpha_post),
        cv_lower = 1.0 / sqrt(alpha_high),
        cv_upper = 1.0 / sqrt(alpha_low)
    )
end

function recommend_ml_model(abc_class::String, v_class::String, zero_ratio::Float64)::Symbol
    zero_ratio == 1.0 && return :DeadStock_NoModel
    is_high_value = (abc_class == "A" || abc_class == "B")
    if v_class == "V1"
        return is_high_value ? :LightGBM_SARIMAX : :SimpleExponentialSmoothing
    elseif v_class == "V2"
        return is_high_value ? :XGBoost_Poisson : :MovingAverage_SafetyStock
    elseif v_class == "V3"
        return is_high_value ? :DeepAR_Probabilistic : :Croston_SBA
    else
        return zero_ratio >= 0.70 ? (is_high_value ? :ZeroInflated_Bayesian_NN : :Poisson_BaseStock_Policy) :
                                    (is_high_value ? :DeepAR_Quantile_LSTM : :Bootstrap_Willemain)
    end
end

function classify_inventory!(df::DataFrame; config::SPAREConfig = SPAREConfig())::DataFrame
    sort!(df, :Revenue, rev=true)
    total_rev = sum(df.Revenue)
    n_rows = nrow(df)

    abc_classes   = Vector{String}(undef, n_rows)
    cv_raw_vals   = Vector{Float64}(undef, n_rows)
    cv_vals       = Vector{Float64}(undef, n_rows)
    cv_upper_vals = Vector{Float64}(undef, n_rows)
    zero_ratios   = Vector{Float64}(undef, n_rows)
    v_classes     = Vector{String}(undef, n_rows)
    boundary_risk = Vector{Bool}(undef, n_rows)
    spare_grids   = Vector{String}(undef, n_rows)
    ml_routes     = Vector{Symbol}(undef, n_rows)

    cum_rev = 0.0
    for i in 1:n_rows
        cum_rev += df.Revenue[i]
        rev_share = total_rev > 0.0 ? (cum_rev / total_rev) : 1.0
        abc = rev_share <= config.abc_a ? "A" : (rev_share <= config.abc_b ? "B" : "C")
        abc_classes[i] = abc

        demands = df.DemandHistory[i]
        n_dem = length(demands)
        avg_d = n_dem > 0 ? mean(demands) : 0.0
        z_ratio = n_dem > 0 ? count(==(0), demands) / n_dem : 1.0
        zero_ratios[i] = round(z_ratio, digits=4)

        cv_raw = (avg_d > 0.0 && n_dem > 1) ? (std(demands) / avg_d) : 0.0
        cv_raw_vals[i] = round(cv_raw, digits=4)

        cv_val = calculate_cv(demands; config=config)
        cv_vals[i] = round(cv_val, digits=4)

        bounds = calculate_bayesian_bounds(demands; config=config)
        cv_up = max(cv_val, bounds.cv_upper)
        cv_upper_vals[i] = round(cv_up, digits=4)

        v_zone, current_fence = cv_val <= config.cv_low ? ("V1", config.cv_low) :
                                cv_val <= config.cv_med ? ("V2", config.cv_med) :
                                cv_val <= config.cv_high ? ("V3", config.cv_high) : ("V4", Inf)
        v_classes[i] = v_zone
        boundary_risk[i] = (cv_up > current_fence)
        spare_grids[i]   = string(abc, "-", v_zone)
        ml_routes[i]     = recommend_ml_model(abc, v_zone, z_ratio)
    end

    df.ABC_Class     = abc_classes
    df.ZeroRatio     = zero_ratios
    df.CV_Raw        = cv_raw_vals
    df.CV            = cv_vals
    df.CV_Upper95    = cv_upper_vals
    df.V_Class       = v_classes
    df.BoundaryRisk  = boundary_risk
    df.SPARE_Grid    = spare_grids
    df.RoutedMLModel = ml_routes
    return df
end

function generate_grid_summary(df::DataFrame; config::SPAREConfig = SPAREConfig())::DataFrame
    summary_df = DataFrame(
        ABC_Class = ["A", "B", "C"],
        V1 = zeros(Int, 3),
        V2 = zeros(Int, 3),
        V3 = zeros(Int, 3),
        V4 = zeros(Int, 3)
    )

    abc_map = Dict("A" => 1, "B" => 2, "C" => 3)
    for row in eachrow(df)
        r_idx = get(abc_map, row.ABC_Class, 0)
        if r_idx > 0 && hasproperty(summary_df, Symbol(row.V_Class))
            summary_df[r_idx, Symbol(row.V_Class)] += 1
        end
    end

    rename!(summary_df,
        :V1 => "V1 (CV <= $(config.cv_low))",
        :V2 => "V2 (CV <= $(config.cv_med))",
        :V3 => "V3 (CV <= $(config.cv_high))",
        :V4 => "V4 (CV > $(config.cv_high))"
    )

    return summary_df
end

function export_ml_features(df::DataFrame)::DataFrame
    abc_num = map(c -> c == "A" ? 3 : (c == "B" ? 2 : 1), df.ABC_Class)
    v_num   = map(v -> parse(Int, String(v)[2:end]), df.V_Class)
    return DataFrame(
        ItemID           = df.ItemID,
        LogRevenue       = round.(log1p.(Float64.(df.Revenue)), digits=4),
        MeanDemand       = round.(map(mean, df.DemandHistory), digits=4),
        ZeroRatio        = df.ZeroRatio,
        HybridBayesianCV = df.CV,
        CV_Upper95       = df.CV_Upper95,
        BoundaryRiskInt  = Int.(df.BoundaryRisk),
        ABC_Encoded      = abc_num,
        V_Encoded        = v_num,
        TargetMLRoute    = df.RoutedMLModel
    )
end

end # module
