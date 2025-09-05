package net.finmath.montecarlo.assetderivativevaluation.risk;

import java.util.Arrays;
import net.finmath.montecarlo.assetderivativevaluation.risk.RiskCalculator;

public final class LossArrayCalculator {
	private LossArrayCalculator() {}

	/**
	 * VaR for raw loss samples (higher = worse).
	 */
	public static double computeVaR(final double[] losses, final double alpha) {
		if (losses == null) {
			throw new NullPointerException("losses must not be null");
		}
		RiskCalculator.validateAlpha(alpha);
		if (losses.length == 0) {
			throw new IllegalArgumentException("losses must have length > 0");
		}
		// Upper-tail VaR at (1 - alpha)
		double[] copy = Arrays.copyOf(losses, losses.length);
		Arrays.sort(copy); // ascending
		int n = copy.length;
		// index for p-th percentile with p = (1 - alpha)
		double p = 1.0 - alpha;
		int idx = (int)Math.ceil(p * n) - 1;
		idx = Math.max(0, Math.min(n - 1, idx));
		return copy[idx];
	}

	/**
	 * CVaR for raw loss samples (higher = worse).
	 */
	public static double computeCVaR(final double[] losses, final double alpha) {
		if (losses == null) {
			throw new NullPointerException("losses must not be null");
		}
		RiskCalculator.validateAlpha(alpha);
		if (losses.length == 0) {
			throw new IllegalArgumentException("losses must have length > 0");
		}
		final double var = computeVaR(losses, alpha);
		double sum = 0.0;
		int cnt = 0;
		for (double v : losses) {
			if (v >= var) {
				sum += v;
				cnt++;
			}
		}
		if (cnt == 0) {
			return var; // conservative fallback
		}
		return sum / cnt;
	}
}
