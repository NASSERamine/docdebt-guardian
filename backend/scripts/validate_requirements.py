"""Validate each functional requirement (BF1-BF20) of DocDebt Guardian by its automated tests.

Usage (from backend/, virtual environment active):
    python scripts/validate_requirements.py            # every requirement
    python scripts/validate_requirements.py BF7 BF12   # some of them

No model, no network, no GitHub: the tests use a scripted model and a fake GitHub.
Prints one line per requirement (OK / FAILED) and exits with 1 if any fails.
"""

from __future__ import annotations

import subprocess
import sys

REQUIREMENTS: dict[str, tuple[str, list[str]]] = {
    "BF1": (
        "Configure the provider, the model and the API key (kept, never shown)",
        [
            "test_the_application_offers_the_four_research_api_models_luna_first",
            "test_keys_are_encrypted_and_only_their_hint_is_kept",
            "test_a_spec_carries_the_key_and_the_prices",
            "test_saving_tests_the_connection_and_never_returns_the_key",
            "test_deleting_the_configuration",
            "test_a_repository_is_reviewed_with_the_configuration_of_its_first_user",
            "test_a_review_uses_the_model_saved_on_the_model_page",
        ],
    ),
    "BF2": (
        "Test the connection before saving; a refused key saves nothing",
        ["test_a_failed_test_saves_nothing", "test_saving_tests_the_connection_and_never_returns_the_key"],
    ),
    "BF3": (
        "Review every pull request, limited to the Java methods it changes",
        [
            "test_valid_pull_request_opened_is_dispatched",
            "test_pull_request_closed_is_ignored",
            "test_extract_touched_units_from_api_fetched_content",
            "test_diff_changed_lines",
            "test_extractor_from_diff",
            "test_missing_signature_is_rejected",
            "test_wrong_signature_is_rejected",
        ],
    ),
    "BF4": (
        "One detection on the whole Javadoc: exact rules OR the model, with an explanation",
        [
            "test_an_exact_rule_is_debt_even_when_the_model_says_complete",
            "test_the_model_alone_can_find_debt_with_its_explanation",
            "test_no_rule_and_a_complete_verdict_proposes_nothing",
            "test_an_explanation_without_exact_quotes_is_kept_without_highlights",
            "test_one_model_one_vote_no_threshold_context_always_on",
            "test_without_a_model_only_the_exact_rules_run",
            "test_a_failing_model_leaves_the_exact_rules_and_says_so",
            "test_auditor_rules",
            "test_auditor_warnings_never_count_as_debt",
        ],
    ),
    "BF5": (
        "Retrieved context on every analysis: siblings + hybrid top-k, leak guard, deterministic",
        [
            "test_siblings_are_the_three_closest_documented_methods_of_the_file",
            "test_rrf_fuses_ranks_and_breaks_ties_by_id",
            "test_hybrid_search_uses_bm25_and_vectors_with_the_leak_guard",
            "test_bm25_alone_when_no_embedder",
            "test_same_index_same_method_same_context",
            "test_context_text_names_each_source_and_stays_within_budget",
            "test_context_is_given_to_the_model_as_reference_only",
            "test_tokenize_splits_identifiers_and_bm25_ranks_relevant_documents",
            "test_chroma_index_answers_like_the_exact_index",
        ],
    ),
    "BF6": (
        "A field-by-field fix (Replacement), verified before it is shown",
        [
            "test_rename_is_deterministic_and_the_model_only_judges",
            "test_a_cited_semantic_debt_is_judged_and_corrected",
            "test_post_validation_reverts_invented_identifiers_and_cosmetic_edits",
            "test_spans_patch_keeps_untouched_lines_byte_identical",
            "test_a_missing_param_is_written_and_a_failed_write_is_retried_once",
            "test_a_write_the_model_never_answers_fails_the_structural_check",
            "test_repair_failure_completes_without_pausing",
        ],
    ),
    "BF7": (
        "Regeneration on demand, attached to the same proposal",
        [
            "test_regenerate_then_accept_the_regeneration",
            "test_accepting_a_regeneration_that_does_not_exist_is_refused",
            "test_slash_commands_regenerate_and_accept_regenerated",
            "test_api_regenerate_and_decide_with_an_edit",
            "test_repairer_retries_with_feedback_then_succeeds",
            "test_repairer_fails_after_max_attempts_and_never_proposes",
        ],
    ),
    "BF8": (
        "A PR comment: problem, explanation, diff, short id; never posted twice",
        [
            "test_review_pull_request_posts_one_proposal_with_a_short_id",
            "test_an_exact_rule_is_debt_even_when_the_model_says_complete",
            "test_clean_method_produces_no_comment",
            "test_reviewing_the_same_pr_again_does_not_repost_a_pending_proposal",
        ],
    ),
    "BF9": (
        "Check run: success / neutral / failure; a missing permission never breaks a review",
        [
            "test_check_run_conclusions",
            "test_repo_policy_is_only_the_merge_block",
            "test_a_missing_checks_permission_never_breaks_a_review",
            "test_no_rule_and_a_complete_verdict_proposes_nothing",
        ],
    ),
    "BF10": (
        "Accept or reject; a rejection needs its reason; bot comments are never commands",
        [
            "test_valid_slash_command_from_a_user_is_dispatched",
            "test_bot_authored_comment_is_never_dispatched_even_with_a_valid_command",
            "test_a_reject_command_without_reason_asks_for_one",
            "test_api_regenerate_and_decide_with_an_edit",
            "test_dashboard_accept_runs_the_decision_and_returns_its_outcome",
            "test_reject_makes_no_commit",
            "test_unknown_short_id_gets_a_friendly_reply_no_crash",
            "test_dashboard_decision_on_unknown_or_resolved_proposal",
            "test_structural_debt_pauses_then_accept_or_reject",
        ],
    ),
    "BF11": (
        "Commit the accepted version, with its safeguards",
        [
            "test_accept_commits_the_fix_and_only_the_fix",
            "test_accept_after_file_changed_on_github_is_reported_not_crashed",
            "test_accept_when_github_rejects_the_commit_is_reported_not_crashed",
            "test_double_accept_is_a_no_op_the_second_time",
            "test_dashboard_then_comment_on_the_same_proposal_commits_once",
            "test_decision_on_a_proposal_whose_graph_run_is_gone_expires_it_cleanly",
            "test_apply_to_file_writes_and_refuses_stale_files",
        ],
    ),
    "BF12": (
        "Accept with an edit, checked by the exact rules, added to the knowledge base",
        [
            "test_accept_with_an_edit_is_checked_by_the_exact_rules",
            "test_edited_text_rules",
            "test_api_regenerate_and_decide_with_an_edit",
        ],
    ),
    "BF13": (
        "Sign in with GitHub",
        [
            "test_login_redirects_to_github_with_a_state",
            "test_callback_refuses_a_wrong_state_or_a_bad_code",
            "test_everything_needs_a_session_in_sign_in_mode",
            "test_local_mode_has_no_sign_in",
            "test_remove_a_repository_and_sign_out",
        ],
    ),
    "BF14": (
        "Add a verified repository (exists, writable, App installed)",
        [
            "test_parse_repo_ref",
            "test_add_a_repository_by_name_or_url",
            "test_sign_in_imports_only_the_repositories_the_user_can_push_to",
            "test_a_user_sees_and_acts_only_on_their_repositories",
        ],
    ),
    "BF15": (
        "Index the knowledge base (button, and every push to the default branch)",
        [
            "test_rebuild_indexes_only_debt_free_documented_methods_tests_and_readme",
            "test_rebuild_keeps_validated_chunks",
            "test_a_push_updates_only_the_changed_files",
            "test_a_push_to_the_default_branch_reindexes_the_changed_files",
            "test_health_scan_indexes_the_base_and_reports_progress",
        ],
    ),
    "BF16": (
        "Dashboard and history: search, filters, pagination, own repositories only",
        [
            "test_api_summary_reflects_store_state",
            "test_api_proposals_lists_recent",
            "test_api_usage_returns_a_series",
            "test_search_filters_by_status_repo_and_text",
            "test_search_paginates_newest_first_with_a_total",
            "test_search_by_date_range",
            "test_api_search_filters_and_validation",
            "test_a_user_sees_and_acts_only_on_their_repositories",
        ],
    ),
    "BF17": (
        "Proposal page: explanation, three versions, retrieved excerpts, links",
        [
            "test_proposal_details_show_the_debt_the_method_and_the_proposed_fix",
            "test_proposal_details_are_none_when_the_run_is_gone",
            "test_api_regenerate_and_decide_with_an_edit",
        ],
    ),
    "BF18": ('"To review" queue of every pending proposal', ["test_review_queue_costs_and_repo_settings_routes"]),
    "BF19": (
        "Scan now over the open pull requests, with live progress",
        [
            "test_scan_reviews_every_open_pr_and_survives_a_failing_one",
            "test_api_scan_starts_in_the_background_and_reports_its_state",
            "test_api_scan_refusals",
            "test_store_allows_one_running_scan_per_repository",
            "test_run_scan_records_the_outcome_or_the_failure",
            "test_a_decided_proposal_is_counted_as_decided_not_as_a_failed_repair",
        ],
    ),
    "BF20": (
        "Observed precision, reject reasons, costs and documentation health",
        [
            "test_precision_by_model_and_committed_mechanism",
            "test_the_cost_of_a_proposal_is_recorded",
            "test_review_queue_costs_and_repo_settings_routes",
            "test_measure_counts_methods_documentation_and_debt",
            "test_health_scan_indexes_the_base_and_reports_progress",
            "test_cost_cap_skips_the_review_and_says_why",
        ],
    ),
}


def run(requirement: str) -> bool:
    title, tests = REQUIREMENTS[requirement]
    expression = " or ".join(sorted(set(tests)))
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-k", expression, "tests"],
        capture_output=True,
        text=True,
    )
    tail = (result.stdout.strip().splitlines() or [""])[-1]
    ok = result.returncode == 0 and "passed" in tail and "failed" not in tail
    print(f"{requirement:5} {'OK    ' if ok else 'FAILED'} {title}  [{tail}]")
    if not ok:
        print(result.stdout[-2000:])
    return ok


def main() -> None:
    wanted = [a.upper() for a in sys.argv[1:]] or list(REQUIREMENTS)
    unknown = [w for w in wanted if w not in REQUIREMENTS]
    if unknown:
        sys.exit(f"Unknown requirement(s): {unknown}. Known: {list(REQUIREMENTS)}")
    results = [run(w) for w in wanted]
    print(f"\n{sum(results)}/{len(results)} functional requirements validated.")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
