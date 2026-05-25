"""Smoke tests: validate action-candidate schemas and meta-action enumeration."""
from rlwa.utils.schemas import ActionCandidate
from rlwa.agents.action_space import META_ACTIONS
from rlwa.rl.featurize import candidate_text


def test_candidate_to_browsergym_click():
    c = ActionCandidate(action_type="click", bid="42", p=0.9, rationale="r")
    s = c.to_browsergym_action()
    assert "click(" in s and "42" in s


def test_candidate_to_browsergym_type():
    c = ActionCandidate(action_type="fill", bid="5", text="hello", p=0.5, rationale="r")
    s = c.to_browsergym_action()
    assert "fill(" in s.lower()
    assert "hello" in s


def test_meta_actions_present():
    assert "ABSTAIN" in META_ACTIONS
    assert "REPORT_INFEASIBLE" in META_ACTIONS
    assert len(META_ACTIONS) == 2


def test_candidate_text_nonempty():
    c = ActionCandidate(action_type="click", bid="1", p=0.1, rationale="hi")
    assert candidate_text(c).strip() != ""
