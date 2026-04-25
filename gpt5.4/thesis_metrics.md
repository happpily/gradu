# Thesis Metrics Mapping

This project now covers the main evaluation indicators required by the thesis brief.

## Metrics required by the image and their code mapping

1. Spectral efficiency
   - Code fields: `sum_rate`, `avg_spectral_efficiency`
   - Meaning: network total spectral efficiency and average spectral efficiency per link
   - Formula:
     - `SINR_i = S_i / (I_i + sigma^2)`
     - `R_i = log2(1 + SINR_i)`
     - `sum_rate = sum_i R_i`

2. Communication reliability / QoS
   - Code fields: `success_ratio`, `outage_ratio`, `min_rate`, `p5_rate`
   - Meaning:
     - `success_ratio`: fraction of links meeting the QoS threshold
     - `outage_ratio`: fraction of links below the threshold
     - `min_rate`: worst-link throughput
     - `p5_rate`: 5th percentile throughput

3. SINR quality
   - Code fields: `avg_sinr_db`, `min_sinr_db`
   - Meaning: average and worst-case received SINR in dB

4. BER
   - Code fields: `avg_ber`, `max_ber`
   - Current approximation:
     - uncoded BPSK BER approximation
     - `BER_i = 0.5 * erfc(sqrt(SINR_i))`
   - Note:
     - This is a communication-theory approximation derived from SINR.
     - If the thesis later requires a specific modulation or coding scheme, this formula can be replaced.

5. Energy efficiency
   - Code field: `energy_efficiency`
   - Meaning: spectral efficiency gained per unit transmit power
   - Formula:
     - `energy_efficiency = sum_rate / sum_i p_i`

6. Fairness
   - Code field: `jain_fairness`
   - Meaning: whether throughput is evenly distributed across links
   - Formula:
     - `J = (sum_i R_i)^2 / (N * sum_i R_i^2)`

7. Power usage
   - Code field: `mean_power`
   - Meaning: average transmit power level used by the algorithm

8. Improvement over baselines
   - Output section: `Improvement of GAT over baselines`
   - Meaning: whether the learned method improves spectral efficiency over traditional methods
   - Current baselines:
     - `Max Power`
     - `Equal Power`
     - `Random`
     - `WMMSE`

## Comparison groups already included

The code now evaluates:

- Multiple algorithm groups:
  - `GAT (Ours)`
  - `Max Power`
  - `Equal Power`
  - `Random`
  - `WMMSE`

- Multiple environment groups:
  - `Light`
  - `Medium`
  - `Heavy`

This satisfies the "3 groups or more comparative experiments" requirement.

## Recommended thesis wording

You can describe the evaluation section like this:

"The proposed GNN-based power allocation method is evaluated using spectral efficiency, QoS satisfaction ratio, outage ratio, SINR, BER, energy efficiency, and fairness. The proposed method is compared with Max Power, Equal Power, Random allocation, and WMMSE under light, medium, and heavy interference scenarios."
