package net.finmath.montecarlo.assetderivativevaluation.risk;

import java.util.Arrays;
import java.util.List;
import net.finmath.montecarlo.assetderivativevaluation.AssetModelMonteCarloSimulationModel;
import net.finmath.montecarlo.assetderivativevaluation.products.AbstractAssetMonteCarloProduct;
import net.finmath.stochastic.RandomVariable;

import net.finmath.montecarlo.assetderivativevaluation.risk.LossArrayCalculator;
import net.finmath.montecarlo.assetderivativevaluation.risk.ProductRiskCalculator;

/**
 Risk metrics based on Monte Carlo samples.

 Conventions used here
 Loss variable:Values represent losses (higher = worse).
 Tail probability alpha:alpha is the size of the worst tail (e.g., 0.05 for the worst 5%).
 VaR(alpha): Quantile at (1 - alpha). Example: VaR(5%) = 95th percentile of loss.
 CVaR(alpha): Average loss in the worst alpha tail (values >= VaR(alpha)).

 Typical usage:

 RandomVariable losses = ...; // e.g. simulated portfolio loss
 double var5  = RiskCalculator.computeVaR(losses, 0.05);
 double cvar5 = RiskCalculator.computeCVaR(losses, 0.05);

 You can also aggregate multiple instruments via {@link #aggregateLosses(RandomVariable...)}.
 */


public final class RiskCalculator {
	private RiskCalculator() {}

	/* ********************************************************************************************
	 * Public API — RandomVariable inputs
	 * ********************************************************************************************/

	/**
	 * Compute Value-at-Risk at tail probability alpha (e.g., alpha=0.05 ⇒ 5% VaR).
	 * Assumes the input is a loss random variable (larger = worse).
	 */
	public static double computeVaR(final RandomVariable losses, final double alpha) {
		if (losses == null) {
			throw new NullPointerException("losses must not be null");
		}
		validateAlpha(alpha);
		// Worst alpha tail ⇒ upper-tail quantile = (1 - alpha)
		return losses.getQuantile(1.0 - alpha);
		// If your finmath build defines quantiles differently, adjust this line accordingly.
	}

	/**
	 * Compute Conditional VaR (Expected Shortfall) at tail probability alpha:
	 * the average of losses that are >= VaR(alpha).
	 */
	public static double computeCVaR(final RandomVariable losses, final double alpha) {
		if (losses == null) {
			throw new NullPointerException("losses must not be null");
		}
		validateAlpha(alpha);

		final double var = computeVaR(losses, alpha);

		// Use values to form the tail and average it.
		final double[] x = losses.getRealizations();
		if (x == null || x.length == 0) {
			throw new IllegalArgumentException("losses has no sample values.");
		}

		double sumTail = 0.0;
		int countTail = 0;
		for (double v : x) {
			if (v >= var) {
				sumTail += v;
				countTail++;
			}
		}

		// Fallback: if numerical ties produce 0 tail count, approximate by expected tail size
		if (countTail == 0) {
			// expected tail size ≈ alpha * N; protect against division by zero
			int expected = Math.max(1, (int)Math.round(alpha * x.length));
			return var; // conservative fallback (or sumTail / expected if you prefer)
		}
		return sumTail / countTail;
	}

	/**
	 * Sum multiple loss random variables into a single portfolio loss.
	 */
	public static RandomVariable aggregateLosses(final RandomVariable... losses) {
		if (losses == null || losses.length == 0) {
			throw new IllegalArgumentException("At least one RandomVariable is required.");
		}
		if (losses[0] == null) {
			throw new NullPointerException("losses[0] must not be null");
		}
		RandomVariable total = losses[0];
		for (int i = 1; i < losses.length; i++) {
			if (losses[i] == null) {
				throw new NullPointerException("losses[" + i + "] must not be null");
			}
			total = total.add(losses[i]);
		}
		return total;
	}

	/* ********************************************************************************************
	 * Convenience overloads — arrays, portfolio lists, and model/product helpers
	 * ********************************************************************************************/

	/**
	 * VaR for raw loss samples (higher = worse).
	 * @deprecated Use {@link LossArrayCalculator#computeVaR(double[], double)} instead.
	 */
	@Deprecated
	public static double computeVaR(final double[] losses, final double alpha) {
		return LossArrayCalculator.computeVaR(losses, alpha);
	}

	/**
	 * CVaR for raw loss samples (higher = worse).
	 * @deprecated Use {@link LossArrayCalculator#computeCVaR(double[], double)} instead.
	 */
	@Deprecated
	public static double computeCVaR(final double[] losses, final double alpha) {
		return LossArrayCalculator.computeCVaR(losses, alpha);
	}

	/**
	 * Convenience: compute VaR from a simulated product by interpreting payoff as P&L and converting to loss.
	 * If your payoff is already a loss, pass it directly to {@link #computeVaR(RandomVariable, double)}.
	 *
	 * @param evaluationTime    usually the product's maturity or reporting time (e.g. 0.0 for present value)
	 * @deprecated Use {@link ProductRiskCalculator#computeVaRFromProduct(AssetModelMonteCarloSimulationModel, AbstractAssetMonteCarloProduct, double, double)} instead.
	 */
	@Deprecated
	public static double computeVaRFromProduct(
			final AssetModelMonteCarloSimulationModel model,
			final AbstractAssetMonteCarloProduct product,
			final double evaluationTime,
			final double alpha
	) throws Exception {
		return ProductRiskCalculator.computeVaRFromProduct(model, product, evaluationTime, alpha);
	}

	/**
	 * Convenience: compute CVaR from a simulated product by interpreting payoff as P&L and converting to loss.
	 * @deprecated Use {@link ProductRiskCalculator#computeCVaRFromProduct(AssetModelMonteCarloSimulationModel, AbstractAssetMonteCarloProduct, double, double)} instead.
	 */
	@Deprecated
	public static double computeCVaRFromProduct(
			final AssetModelMonteCarloSimulationModel model,
			final AbstractAssetMonteCarloProduct product,
			final double evaluationTime,
			final double alpha
	) throws Exception {
		return ProductRiskCalculator.computeCVaRFromProduct(model, product, evaluationTime, alpha);
	}

	/**
	 * Convenience: aggregate a list of loss RandomVariables into one and return VaR.
	 */
	public static double computePortfolioVaR(final List<RandomVariable> lossList, final double alpha) {
		if (lossList == null) {
			throw new NullPointerException("lossList must not be null");
		}
		if (lossList.isEmpty()) {
			throw new IllegalArgumentException("lossList must not be empty");
		}
		return computeVaR(aggregateLosses(lossList.toArray(RandomVariable[]::new)), alpha);
	}

	/**
	 * Convenience: aggregate a list of loss RandomVariables into one and return CVaR.
	 */
	public static double computePortfolioCVaR(final List<RandomVariable> lossList, final double alpha) {
		if (lossList == null) {
			throw new NullPointerException("lossList must not be null");
		}
		if (lossList.isEmpty()) {
			throw new IllegalArgumentException("lossList must not be empty");
		}
		return computeCVaR(aggregateLosses(lossList.toArray(RandomVariable[]::new)), alpha);
	}

	/* ********************************************************************************************
	 * Helpers
	 * ********************************************************************************************/

	/* package */ static void validateAlpha(final double alpha) {
		if (!(alpha > 0.0 && alpha < 1.0)) {
			throw new IllegalArgumentException("alpha must be in (0,1). Given: " + alpha);
		}
	}
}
