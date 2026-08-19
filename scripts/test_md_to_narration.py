#!/usr/bin/env python3
"""Tests for `md-to-narration.py`.

    scripts/test_md_to_narration.py

Every case here is one thing a listener heard wrong. Technical prose is where this script
earns its keep and where its failures are least visible on the page: a stray period inside a
number is a sentence boundary the engine believes, a silent symbol turns an approximation
into an equality, and an unexpanded unit is spelled letter by letter.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "md_to_narration", Path(__file__).with_name("md-to-narration.py")
)
narration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(narration)

clean = narration.clean_inline
convert = narration.convert


class Numbers(unittest.TestCase):
    def test_decimals_and_versions_survive(self):
        self.assertEqual(clean("Latency rose to 12.4 seconds."), "Latency rose to 12.4 seconds.")
        self.assertEqual(clean("Shipped in v1.2.3."), "Shipped in v1.2.3.")

    def test_ranges_are_read_as_ranges(self):
        self.assertEqual(clean("Temperatures of 20–25 degrees"), "Temperatures of 20 to 25 degrees")
        self.assertEqual(clean("6-12 months of demand"), "6 to 12 months of demand")
        self.assertEqual(clean("over FY2024–25"), "over FY2024 to 2025")

    def test_scientific_notation(self):
        self.assertEqual(clean("fell to 1.5e-3"), "fell to 1.5 times ten to the minus 3")
        self.assertEqual(clean("about 10^-3"), "about ten to the minus 3")
        self.assertEqual(clean("2.1 × 10^22 FLOPs"), "2.1 times ten to the 22 FLOPs")

    def test_intervals_are_not_lists(self):
        self.assertEqual(clean("95% CI [1.2, 3.4]"), "95 percent CI 1.2 to 3.4")

    def test_comparisons_are_spoken(self):
        self.assertEqual(clean("p < 0.05"), "p less than 0.05")
        self.assertEqual(clean("N = 42 per arm"), "N equals 42 per arm")
        self.assertEqual(clean("Supply = Demand + Net Exports"),
                         "Supply equals Demand plus Net Exports")

    def test_an_equation_wrapped_across_lines_keeps_its_operator(self):
        self.assertEqual(clean("Generation = Consumption +"), "Generation equals Consumption plus")

    def test_iso_dates(self):
        self.assertEqual(clean("from 2021-03-04 to then"), "from March 4, 2021 to then")

    def test_magnitudes_and_percentage_points(self):
        self.assertEqual(clean("7B parameters"), "7 billion parameters")
        self.assertEqual(clean("a gain of 17.6 pp"), "a gain of 17.6 percentage points")


class Units(unittest.TestCase):
    def test_rates_are_rates(self):
        self.assertEqual(clean("Throughput was 3.2 GB/s"), "Throughput was 3.2 gigabytes per second")
        self.assertEqual(clean("60 %/yr"), "60 percent per year")

    def test_an_alternation_is_not_a_rate(self):
        self.assertEqual(clean("every bus/node on the grid"), "every bus or node on the grid")

    def test_number_agreement(self):
        self.assertEqual(clean("consume 300 MW of power"), "consume 300 megawatts of power")
        self.assertEqual(clean("plug in 1 MW for an hour"), "plug in 1 megawatt for an hour")
        self.assertEqual(clean("a 4.4 GW natural gas plant"), "a 4.4 gigawatt natural gas plant")

    def test_a_gloss_defines_the_abbreviation_it_names(self):
        self.assertEqual(clean("measured in megawatts (MW) for most units"),
                         "measured in megawatts (MW) for most units")

    def test_a_bare_power_unit_is_still_spoken(self):
        self.assertEqual(clean("50 dollars per MWh"), "50 dollars per megawatt hour")


class Currency(unittest.TestCase):
    def test_a_magnitude_suffix_keeps_its_fraction(self):
        self.assertEqual(clean("Costs were ~$1.5M over the year"),
                         "Costs were about 1.5 million dollars over the year")

    def test_a_price_range_carries_the_unit_at_both_ends(self):
        self.assertEqual(clean("anywhere from $10-150 per MWh"),
                         "anywhere from 10 dollars to 150 dollars per megawatt hour")

    def test_a_floor_is_not_an_addition(self):
        self.assertEqual(clean("like $100M+ machines"), "like 100 million dollars or higher machines")

    def test_cents_and_attributive_use_still_work(self):
        self.assertEqual(clean("credit $12.50 to revenue"), "credit 12 dollars 50 cents to revenue")
        self.assertEqual(clean("a $12 platform fee"), "a 12 dollar platform fee")


class Abbreviations(unittest.TestCase):
    def test_expansion(self):
        self.assertEqual(clean("energy (e.g. power), agriculture"),
                         "energy (for example power), agriculture")
        self.assertEqual(clean("the DA market vs. the RT market"),
                         "the DA market versus the RT market")
        self.assertEqual(clean("cf. Fig. 2 and Table 1"), "compare Figure 2 and Table 1")
        self.assertEqual(clean("vol. 33, no. 2, pp. 2175–2183"),
                         "volume 33, number 2, pages 2175 to 2183")

    def test_an_abbreviation_can_close_a_line(self):
        self.assertEqual(clean("the Western vs."), "the Western versus")

    def test_initials_collapse_only_in_runs(self):
        self.assertEqual(clean("Dr. J. R. R. Tolkien, Ph.D."), "Doctor J R R Tolkien, PhD")
        # A lone capital before a full stop ends a sentence; stripping it welds two together.
        self.assertEqual(clean("There's 10 MWh of demand at A. The model solves it."),
                         "There's 10 megawatt hours of demand at A. The model solves it.")


class Maths(unittest.TestCase):
    def test_an_inline_span_is_verbalised(self):
        self.assertEqual(clean(r"where $\alpha = 0.9$ and $\beta \in [0, 1]$"),
                         "where alpha equals 0.9 and beta in the range 0 to 1")

    def test_operators_and_accents(self):
        self.assertEqual(clean(r"$\hat{\theta} = \arg\max_\theta \sum_i \log p(x_i)$"),
                         "theta hat equals arg max over theta the sum over i of log p(x i)")

    def test_display_maths_is_shown_not_spoken(self):
        out = convert("Before.\n\n$$\n\\sum_{i=1}^{N} x_i\n$$\n\nAfter.\n")
        self.assertEqual(out, "Before.\n\nAfter.\n")

    def test_glyphs_outside_a_span(self):
        self.assertEqual(clean("d ≈ 0.42, Δ ≤ 5% and r ~ 0.3"),
                         "d about 0.42, delta at most 5 percent and r about 0.3")
        self.assertEqual(clean("The α-β trade-off"), "The alpha beta trade off")


class Citations(unittest.TestCase):
    def test_a_semicolon_inside_a_citation_is_not_a_full_stop(self):
        self.assertEqual(clean("(Smith et al., 2021; Zhou & Lee, 2019) reports"),
                         "(Smith and colleagues, 2021, Zhou and Lee, 2019) reports")

    def test_a_semicolon_between_clauses_still_splits(self):
        self.assertEqual(clean("queue age; webhook retries"), "queue age. Webhook retries")

    def test_a_url_is_read_as_its_host(self):
        self.assertEqual(clean("Data at https://doi.org/10.1234/abcd.5678 (accessed)"),
                         "Data at doi.org (accessed)")

    def test_footnotes(self):
        out = convert("A claim.[^1]\n\n[^1]: The replication package.\n")
        self.assertEqual(out, "A claim.\n\nFootnote 1. The replication package.\n")


class Regressions(unittest.TestCase):
    """Behaviour the technical rules must not have disturbed."""

    def test_prose_is_untouched(self):
        self.assertEqual(clean("An ordinary sentence, with a comma."),
                         "An ordinary sentence, with a comma.")

    def test_identifiers_and_code_spans(self):
        self.assertEqual(clean("`PaymentCaptured(provider_transaction_id)` fires"),
                         "Payment captured, provider transaction id fires")
        self.assertEqual(clean("an NP-15 price in CAISO"), "an NP-15 price in CAISO")

    def test_emphasis_and_dashes(self):
        self.assertEqual(clean("**bold** and *italic* — a pause"), "bold and italic, a pause")


if __name__ == "__main__":
    unittest.main(verbosity=2)
