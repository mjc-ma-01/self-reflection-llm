from self_reflection_llm.rl.gdpo import group_decoupled_advantages
from self_reflection_llm.rl.reward import MARK_CONTINUE, MARK_EXPLORE, MARK_REFLECT, compute_score_segmented, split_response_by_markers


def test_split_response_by_markers_prefers_continue_for_final_text():
    text = f"unsafe draft {MARK_REFLECT} reflection {MARK_EXPLORE} alternative {MARK_CONTINUE} final refusal"
    initial, final, has_markers = split_response_by_markers(text)
    assert initial == "unsafe draft"
    assert final == "final refusal"
    assert has_markers is True


def test_segmented_harmful_reward_uses_final_refusal_and_reflection_bonus():
    solution = f"Here is how to do it. {MARK_REFLECT} This is unsafe. {MARK_CONTINUE} I cannot help with that."
    result = compute_score_segmented(
        data_source="reflector_harmful",
        solution_str=solution,
        ground_truth="refuse",
        extra_info={"prompt_type": "harmful", "expected_behavior": "refuse", "question": "bad request"},
    )
    assert result["final_ok"] is True
    assert result["reward_component_correctness"] == 1.0
    assert result["reward_component_reflect"] == 1.0
    assert result["score"] == 1.3


def test_segmented_harmful_reward_penalizes_bad_reflection_outcome():
    solution = f"Here is how to do it. {MARK_REFLECT} This is unsafe. {MARK_CONTINUE} Still, follow these steps."
    result = compute_score_segmented(
        data_source="reflector_harmful",
        solution_str=solution,
        ground_truth="refuse",
        extra_info={"prompt_type": "harmful", "expected_behavior": "refuse", "question": "bad request"},
    )
    assert result["final_ok"] is False
    assert result["reward_component_correctness"] == 0.0
    assert result["reward_component_reflect"] == -1.0
    assert result["score"] == -0.3


def test_gdpo_advantages_decouple_components_by_group():
    advantages = group_decoupled_advantages(
        group_ids=["a", "a", "b", "b"],
        reward_components={
            "reward_component_correctness": [1.0, 0.0, 0.0, 1.0],
            "reward_component_reflect": [1.0, -1.0, 0.0, 0.0],
        },
    )
    assert len(advantages) == 4
    assert round(sum(advantages), 7) == 0.0
    assert advantages[0] > advantages[1]
    assert advantages[3] > advantages[2]
