import math

import pytest
import spacy

from seshat.pipeline.extraction.heuristics_scorer import HeuristicsScorer


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="module")
def scorer(nlp):
    return HeuristicsScorer(nlp=nlp)


class TestScore:
    def test_empty_fields_return_zero(self, scorer: HeuristicsScorer):
        assert scorer.score("", "", "") == 0.0

    def test_score_in_range(self, scorer: HeuristicsScorer):
        s = scorer.score(
            "we decided to use PostgreSQL for session storage",
            "Use PostgreSQL for session storage",
            "The team decided to use PostgreSQL for all persistent data.",
        )
        assert 0.0 <= s <= 1.0

    def test_strong_inputs_score_above_half(self, scorer: HeuristicsScorer):
        s = scorer.score(
            "we decided to use PostgreSQL for session storage",
            "Use PostgreSQL for session storage",
            "The team decided to use PostgreSQL for all persistent data.",
        )
        assert s > 0.5

    def test_weak_inputs_score_below_strong(self, scorer: HeuristicsScorer):
        strong = scorer.score(
            "we decided to use PostgreSQL for session storage",
            "Use PostgreSQL for session storage",
            "The team decided to use PostgreSQL for all persistent data.",
        )
        weak = scorer.score("", "Decision", "We might do something.")
        assert weak < strong

    def test_quote_word_count_saturates(self, scorer: HeuristicsScorer):
        long_quote = " ".join(["word"] * 70)
        saturated = scorer.score(long_quote, "", "")
        over_saturated = scorer.score(long_quote + " extra word", "", "")
        assert math.isclose(saturated, over_saturated)


class TestTitleSpecificity:
    def test_empty_returns_zero(self, scorer: HeuristicsScorer):
        assert scorer._title_specificity("") == 0.0

    def test_whitespace_only_returns_zero(self, scorer: HeuristicsScorer):
        assert scorer._title_specificity("   ") == 0.0

    def test_tech_pattern_with_qualifier_scores_high(self, scorer: HeuristicsScorer):
        s = scorer._title_specificity("Use PostgreSQL for session storage")
        assert s >= 0.8

    def test_tech_pattern_without_qualifier_scores_medium(self, scorer: HeuristicsScorer):
        s = scorer._title_specificity("Use PostgreSQL")
        assert 0.4 <= s <= 0.75

    def test_generic_label_scores_low(self, scorer: HeuristicsScorer):
        s = scorer._title_specificity("Database decision")
        assert s < 0.5

    def test_longer_title_scores_higher_than_shorter(self, scorer: HeuristicsScorer):
        short = scorer._title_specificity("Use Redis")
        long = scorer._title_specificity("Migrate session storage from in-memory cache to Redis")
        assert long > short

    def test_entity_and_qualifier_both_contribute(self, scorer: HeuristicsScorer):
        entity_only = scorer._title_specificity("Use Redis")
        entity_and_qualifier = scorer._title_specificity("Use Redis for session storage")
        assert entity_and_qualifier > entity_only


class TestDirectness:
    def test_empty_returns_zero(self, scorer: HeuristicsScorer):
        assert scorer._directness("") == 0.0

    def test_whitespace_only_returns_zero(self, scorer: HeuristicsScorer):
        assert scorer._directness("   ") == 0.0

    def test_active_with_complement_returns_one(self, scorer: HeuristicsScorer):
        s = scorer._directness("The team chose PostgreSQL.")
        assert math.isclose(s, 1.0)

    def test_hedged_description_penalised(self, scorer: HeuristicsScorer):
        direct = scorer._directness("The team chose PostgreSQL.")
        hedged = scorer._directness("The team might choose PostgreSQL.")
        assert hedged < direct

    def test_passive_voice_penalised(self, scorer: HeuristicsScorer):
        active = scorer._directness("The team chose PostgreSQL.")
        passive = scorer._directness("PostgreSQL was chosen by the team.")
        assert passive < active

    def test_future_tense_penalised(self, scorer: HeuristicsScorer):
        past = scorer._directness("The team chose PostgreSQL.")
        future = scorer._directness("The team will choose PostgreSQL.")
        assert future < past

    def test_no_complement_penalised(self, scorer: HeuristicsScorer):
        with_complement = scorer._directness("The team chose PostgreSQL.")
        without_complement = scorer._directness("The team agreed.")
        assert without_complement < with_complement

    def test_passive_and_hedged_penalised_independently(self, scorer: HeuristicsScorer):
        active_direct = scorer._directness("The team chose PostgreSQL.")
        passive_hedged = scorer._directness("PostgreSQL might be chosen.")
        assert passive_hedged < active_direct * 0.9

    def test_subordinate_hedge_does_not_penalise_main_clause(self, scorer: HeuristicsScorer):
        # "might" is in a subordinate clause, not a child of the root
        main_clause = scorer._directness("We chose PostgreSQL.")
        with_subordinate = scorer._directness("We chose PostgreSQL, though it might need tuning.")
        assert with_subordinate >= main_clause * 0.9


class TestDirectnessEdgeCases:
    def test_would_is_treated_as_hedge(self, scorer: HeuristicsScorer):
        direct = scorer._directness("The team chose PostgreSQL.")
        hedged = scorer._directness("We would use Redis.")
        assert math.isclose(hedged, 0.5)
        assert hedged < direct

    def test_shall_is_treated_as_future(self, scorer: HeuristicsScorer):
        past = scorer._directness("The team chose PostgreSQL.")
        future = scorer._directness("The team shall use Redis.")
        assert math.isclose(future, 0.75)
        assert future < past

    def test_future_and_passive_stack_independently(self, scorer: HeuristicsScorer):
        # future x0.75, passive x0.75, no complement x0.75
        s = scorer._directness("Redis will be used.")
        assert math.isclose(s, 0.75 * 0.75 * 0.75, rel_tol=1e-6)

    def test_hedge_in_second_sentence_does_not_penalise_first(self, scorer: HeuristicsScorer):
        # scorer only looks at the first ROOT; hedge in sentence 2 must be ignored
        assert math.isclose(scorer._directness("We chose Redis. It might need tuning."), 1.0)


class TestTitleSpecificityEdgeCases:
    def test_qualifier_without_entity_scores_partial(self, scorer: HeuristicsScorer):
        # only word count and qualifier contribute — no entity
        s = scorer._title_specificity("Decision for the backend team")
        assert 0.0 < s < scorer._W_TITLE_QUALIFIER + scorer._W_TITLE_WORDS


class TestScoreEdgeCases:
    def test_quote_at_exact_saturation_boundary(self, scorer: HeuristicsScorer):
        quote = " ".join(["word"] * scorer._QUOTE_WORD_SATURATION)
        s = scorer.score(quote, "", "")
        assert math.isclose(s, scorer._W_QUOTE)

    def test_quote_one_word_over_boundary_does_not_exceed_weight(self, scorer: HeuristicsScorer):
        quote = " ".join(["word"] * (scorer._QUOTE_WORD_SATURATION + 1))
        s = scorer.score(quote, "", "")
        assert math.isclose(s, scorer._W_QUOTE)


class TestHasNamedEntity:
    def test_tech_pattern_match(self, scorer: HeuristicsScorer, nlp):
        assert scorer._has_named_entity(nlp("use redis for caching"))

    def test_tech_pattern_case_insensitive(self, scorer: HeuristicsScorer, nlp):
        assert scorer._has_named_entity(nlp("use Redis for caching"))

    def test_propn_match(self, scorer: HeuristicsScorer, nlp):
        assert scorer._has_named_entity(nlp("Alice will lead the project"))

    def test_no_entity_returns_false(self, scorer: HeuristicsScorer, nlp):
        assert not scorer._has_named_entity(nlp("use a database for storage"))

    def test_all_tech_patterns_recognised(self, scorer: HeuristicsScorer, nlp):
        for term in scorer._TECH_PATTERNS:
            assert scorer._has_named_entity(nlp(f"we use {term}")), f"failed for: {term}"


class TestHasQualifier:
    def test_prep_on_root_verb(self, scorer: HeuristicsScorer, nlp):
        assert scorer._has_qualifier(nlp("Use Redis for session storage"))

    def test_noun_root_without_prep_does_not_qualify(self, scorer: HeuristicsScorer, nlp):
        # no prep or advcl — nothing to qualify the noun root
        assert not scorer._has_qualifier(nlp("the database decision"))

    def test_adverbial_clause_on_root_qualifies(self, scorer: HeuristicsScorer, nlp):
        # "when traffic spikes" is an advcl on the root verb "Use"
        assert scorer._has_qualifier(nlp("Use Redis when traffic spikes"))

    def test_no_qualifier(self, scorer: HeuristicsScorer, nlp):
        assert not scorer._has_qualifier(nlp("Use Redis"))

    def test_empty_doc(self, scorer: HeuristicsScorer, nlp):
        assert not scorer._has_qualifier(nlp(""))

    def test_prep_on_noun_root_qualifies(self, scorer: HeuristicsScorer, nlp):
        # "Database choice for session storage" — root is NOUN, prep "for" is a genuine qualifier
        assert scorer._has_qualifier(nlp("Database choice for session storage"))
