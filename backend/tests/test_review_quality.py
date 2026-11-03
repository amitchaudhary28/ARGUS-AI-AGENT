from app.main import (
    clean_ai_review_markdown,
    deduplicate_ai_findings,
    merge_deterministic_findings,
    validate_ai_review_evidence,
)


def test_removes_empty_headings_and_placeholder_bullets():
    review = """
## Confirmed Findings

None identified.

###
-
## Potential Risks
-
"""

    cleaned = clean_ai_review_markdown(review)

    assert "###\n" not in cleaned
    assert "- \n" not in cleaned


def test_deterministic_finding_is_not_hidden_by_ai_review():
    review = """
## Confirmed Findings

None identified.

## Potential Risks

No additional risks identified.
"""

    findings = [
        {
            "severity": "High",
            "description": "Division by zero",
            "line": 3,
            "evidence": "return divide(10, 0)",
            "verification": "Deterministic static check",
            "suggested_fix": "Validate the divisor before division.",
        }
    ]

    merged = merge_deterministic_findings(
        review,
        findings,
        1,
        10,
    )

    assert "Division by zero" in merged
    assert "Source line: 3" in merged
    assert "None identified." not in merged


def test_flags_evidence_that_does_not_match_source_line():
    source_lines = [
        "def divide(a, b):",
        "    return a / b",
        "result = divide(10, 2)",
    ]

    review = """
## Confirmed Findings

### High: Division by zero
- Source line: 2
- Evidence: `return a / 0`
"""

    validated = validate_ai_review_evidence(
        review=review,
        source_lines=source_lines,
        valid_start_line=1,
        valid_end_line=3,
    )

    assert "UNVERIFIED" in validated
    assert "does not match" in validated


def test_accepts_evidence_matching_source_line():
    source_lines = [
        "def divide(a, b):",
        "    return a / b",
        "result = divide(10, 2)",
    ]

    review = """
## Confirmed Findings

### Medium: Division operation
- Source line: 2
- Evidence: `return a / b`
"""

    validated = validate_ai_review_evidence(
        review=review,
        source_lines=source_lines,
        valid_start_line=1,
        valid_end_line=3,
    )

    assert "does not match" not in validated
    assert "UNVERIFIED" not in validated


def test_validates_original_line_number_for_later_chunk():
    source_lines = [
        "first_line = 1",
        "second_line = 2",
        "third_line = 3",
        "fourth_line = 4",
    ]

    review = """
## Confirmed Findings

### Low: Example finding
- Source line: 4
- Evidence: `fourth_line = 4`
"""

    validated = validate_ai_review_evidence(
        review=review,
        source_lines=source_lines,
        valid_start_line=3,
        valid_end_line=4,
    )

    assert "does not match" not in validated
    assert "UNVERIFIED" not in validated


def test_removes_unexpected_ai_review_section():
    review = """
## Confirmed Findings

None identified.

## Potential Risks

None identified.

### Review of the Source Code

- Source Line: 6
- Evidence: divide(10, 0)
"""

    cleaned = clean_ai_review_markdown(review)

    assert "Review of the Source Code" not in cleaned
    assert "## Confirmed Findings" in cleaned
    assert "## Potential Risks" in cleaned


def test_preserves_both_approved_sections():
    review = """
## Confirmed Findings

### High: Division by zero
- Source line: 7
- Evidence: `divide(10, 0)`

## Potential Risks

None identified.
"""

    cleaned = clean_ai_review_markdown(review)

    assert "Division by zero" in cleaned
    assert "divide(10, 0)" in cleaned
    assert "## Confirmed Findings" in cleaned
    assert "## Potential Risks" in cleaned



def test_flags_out_of_range_source_line():
    source_lines = [
        "def calculate():",
        "    return 10",
    ]

    review = """
## Confirmed Findings

### High: Incorrect calculation
- Source line: 100
- Evidence: `return 10`
"""

    validated = validate_ai_review_evidence(
        review=review,
        source_lines=source_lines,
        valid_start_line=1,
        valid_end_line=2,
    )

    assert "UNVERIFIED" in validated
    assert "outside" in validated



def test_duplicate_ai_findings_are_removed():
    review = """
## Confirmed Findings

### High: Division by zero
- Source line: 7
- Evidence: `divide(10, 0)`

### High: Division by zero
- Source line: 7
- Evidence: `divide(10, 0)`

## Potential Risks

None identified.
"""

    cleaned = clean_ai_review_markdown(review)
    deduplicated = deduplicate_ai_findings(cleaned)

    assert deduplicated.count("Division by zero") == 1


def test_missing_evidence_is_unverified():
    source_lines = [
        "def calculate():",
        "    return 10",
    ]

    review = """
## Confirmed Findings

### High: Incorrect calculation
- Source line: 2
"""

    validated = validate_ai_review_evidence(
        review=review,
        source_lines=source_lines,
        valid_start_line=1,
        valid_end_line=2,
    )

    assert "UNVERIFIED" in validated
    assert "no explicit evidence" in validated


def test_repeated_validation_does_not_duplicate_labels():
    source_lines = [
        "def divide(a, b):",
        "    return a / b",
    ]

    review = """
## Confirmed Findings

### High: Division by zero
- Source line: 2
- Evidence: `return a / b`
"""

    first_pass = validate_ai_review_evidence(
        review=review,
        source_lines=source_lines,
        valid_start_line=1,
        valid_end_line=2,
    )

    second_pass = validate_ai_review_evidence(
        review=first_pass,
        source_lines=source_lines,
        valid_start_line=1,
        valid_end_line=2,
    )

    assert second_pass.count("Evidence validation:") == 1
    assert "EVIDENCE_MATCHED" in second_pass


def test_multiple_findings_are_validated_independently():
    source_lines = [
        "def divide(a, b):",
        "    return a / b",
        "result = divide(10, 0)",
    ]

    review = """
## Confirmed Findings

### High: Division operation
- Source line: 2
- Evidence: `return a / b`

### Medium: Incorrect function call
- Source line: 3
- Evidence: `divide(10, 2)`
"""

    validated = validate_ai_review_evidence(
        review=review,
        source_lines=source_lines,
        valid_start_line=1,
        valid_end_line=3,
    )

    assert validated.count("Evidence validation:") == 2
    assert validated.count("EVIDENCE_MATCHED") == 1
    assert validated.count("UNVERIFIED") == 1
