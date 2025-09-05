package net.finmath.montecarlo.assetderivativevaluation.risk;

import net.finmath.montecarlo.assetderivativevaluation.AssetModelMonteCarloSimulationModel;
import net.finmath.montecarlo.assetderivativevaluation.products.AbstractAssetMonteCarloProduct;
import net.finmath.stochastic.RandomVariable;
import net.finmath.montecarlo.assetderivativevaluation.risk.RiskCalculator;

public final class ProductRiskCalculator {
	private ProductRiskCalculator() {}

	/**
	 * Convenience: compute VaR from a simulated product by interpreting payoff as P&L and converting to loss.
	 * If your payoff is already a loss, pass it directly to {@link RiskCalculator#computeVaR(RandomVariable, double)}.
	 *
	 * @param evaluationTime    usually the product's maturity or reporting time (e.g. 0.0 for present value)
	 */
	public static double computeVaRFromProduct(
			final AssetModelMonteCarloSimulationModel model,
			final AbstractAssetMonteCarloProduct product,
			final double evaluationTime,
			final double alpha
	) throws Exception {
		if (model == null) {
			throw new NullPointerException("model must not be null");
		}
		if (product == null) {
			throw new NullPointerException("product must not be null");
		}
		RiskCalculator.validateAlpha(alpha);

		// Interpret product value as P&L and convert to loss = -P&L
		final RandomVariable pnl = product.getValue(evaluationTime, model);
		final RandomVariable loss = pnl.mult(-1.0);
		return RiskCalculator.computeVaR(loss, alpha);
	}

	/**
	 * Convenience: compute CVaR from a simulated product by interpreting payoff as P&L and converting to loss.
	 */
	public static double computeCVaRFromProduct(
			final AssetModelMonteCarloSimulationModel model,
			final AbstractAssetMonteCarloProduct product,
			final double evaluationTime,
			final double alpha
	) throws Exception {
		if (model == null) {
			throw new NullPointerException("model must not be null");
		}
		if (product == null) {
			throw new NullPointerException("product must not be null");
		}
		RiskCalculator.validateAlpha(alpha);

		final RandomVariable pnl = product.getValue(evaluationTime, model);
		final RandomVariable loss = pnl.mult(-1.0);
		return RiskCalculator.computeCVaR(loss, alpha);
	}
}
