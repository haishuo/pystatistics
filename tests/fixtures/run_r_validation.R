#!/usr/bin/env Rscript
#
# R Reference Validation for PyStatistics
#
# Runs lm() on all fixture CSVs and saves results with maximum precision.
#
# IMPORTANT: We fit using raw matrix algebra to match PyStatistics exactly.
# This avoids R's special handling of intercepts and formula semantics.
#
# Run from /mnt/projects/pystatistics:
#   Rscript tests/fixtures/run_r_validation.R
#

library(jsonlite)

options(digits = 22)

fixtures_dir <- "tests/fixtures"
csv_files <- list.files(fixtures_dir, pattern = "\\.csv$", full.names = TRUE)

cat("Running R validation on", length(csv_files), "fixtures...\n\n")

for (csv_file in csv_files) {
    fixture_name <- tools::file_path_sans_ext(basename(csv_file))
    if (grepl("_r_results", fixture_name)) next
    
    cat("Processing:", fixture_name, "\n")
    
    # Read data
    data <- read.csv(csv_file)
    
    # Extract X and y
    y <- as.numeric(data$y)
    X <- as.matrix(data[, setdiff(names(data), "y")])
    
    n <- nrow(X)
    p <- ncol(X)
    
    # Solve via QR decomposition (same as PyStatistics)
    qr_X <- qr(X)
    
    # Coefficients: solve R * beta = Q' * y
    Q <- qr.Q(qr_X)
    R <- qr.R(qr_X)
    
    # beta = solve(R) %*% t(Q) %*% y
    Qty <- t(Q) %*% y
    beta <- backsolve(R, Qty)
    
    # Fitted values and residuals
    fitted <- as.numeric(X %*% beta)
    residuals <- y - fitted
    
    # Residual sum of squares
    rss <- sum(residuals^2)
    
    # Total sum of squares (CENTERED - same as PyStatistics)
    y_mean <- mean(y)
    tss <- sum((y - y_mean)^2)
    
    # R-squared (using centered TSS)
    r_squared <- 1 - rss / tss
    
    # Degrees of freedom
    rank <- qr_X$rank
    df_residual <- n - rank
    
    # Residual standard error
    sigma <- sqrt(rss / df_residual)
    
    # Adjusted R-squared
    adj_r_squared <- 1 - (1 - r_squared) * (n - 1) / df_residual
    
    # Standard errors of coefficients
    # SE = sigma * sqrt(diag((X'X)^-1))
    # Use QR to compute (X'X)^-1 = R^-1 R^-T for numerical stability
    R_inv <- backsolve(R, diag(p))
    XtX_inv <- R_inv %*% t(R_inv)
    se <- sigma * sqrt(diag(XtX_inv))
    
    # t-statistics
    t_stats <- as.numeric(beta) / se
    
    # p-values (two-sided)
    p_values <- 2 * pt(abs(t_stats), df = df_residual, lower.tail = FALSE)
    
    results <- list(
        fixture = fixture_name,
        method = "QR decomposition (matching PyStatistics)",
        
        coefficients = as.numeric(beta),
        standard_errors = as.numeric(se),
        t_statistics = as.numeric(t_stats),
        p_values = as.numeric(p_values),
        
        residuals_head = as.numeric(head(residuals, 10)),
        residuals_tail = as.numeric(tail(residuals, 10)),
        fitted_head = as.numeric(head(fitted, 10)),
        fitted_tail = as.numeric(tail(fitted, 10)),
        
        r_squared = r_squared,
        adj_r_squared = adj_r_squared,
        sigma = sigma,
        df_residual = df_residual,
        
        rss = rss,
        tss = tss,
        
        rank = rank,
        
        residuals_all = as.numeric(residuals),
        fitted_all = as.numeric(fitted)
    )
    
    output_file <- file.path(fixtures_dir, paste0(fixture_name, "_r_results.json"))
    json_str <- toJSON(results, auto_unbox = TRUE, digits = 17, pretty = TRUE)
    writeLines(json_str, output_file)
    
    cat("  ✓ R²:", format(r_squared, digits = 10), "\n")
    cat("  ✓ Saved to:", output_file, "\n\n")
}

cat("✅ R validation complete!\n")