import pickle
from types import SimpleNamespace

from pgm_llm_inference.mpe.client import LLMJsonClient, _attach_domain_scores
from pgm_llm_inference.mpe.bucket import semantic_message_from_response
from pgm_llm_inference.mpe.domain_scoring import build_state_code_map
from pgm_llm_inference.mpe.io import build_alias_map
from pgm_llm_inference.mpe.prompt_builders import build_bucket_prompt
from pgm_llm_inference.mpe.types import (
    BucketResponse,
    BucketSpec,
    DecisionTokenScores,
    DomainScores,
    MessageRow,
    TokenScore,
)
from pgm_llm_inference.models import BayesianNetwork


def _token(token: str, logprob: float, alternatives=()):
    return SimpleNamespace(
        token=token,
        bytes=list(token.encode("utf-8")),
        logprob=logprob,
        top_logprobs=[
            SimpleNamespace(token=value, logprob=score)
            for value, score in alternatives
        ],
    )


def test_attaches_scores_to_each_context_decision() -> None:
    response = (
        '{"decisions":['
        '{"context":{"B":"1"},"selected_value":"B"},'
        '{"context":{"B":"0"},"selected_value":"A"}'
        "]}"
    )
    parsed = {
        "decisions": [
            {"context": {"B": "1"}, "selected_value": "B"},
            {"context": {"B": "0"}, "selected_value": "A"},
        ]
    }
    first_value = response.index('"selected_value":"B"') + len('"selected_value":"')
    second_value = response.index('"selected_value":"A"') + len('"selected_value":"')
    token_logprobs = [
        _token(response[:first_value], -0.01),
        _token(
            "B",
            -0.1,
            (("A", -2.0), ("B", -0.1), ('"A', -9.0), ('"B', -8.0)),
        ),
        _token(response[first_value + 1 : second_value], -0.01),
        _token("A", -0.2, (("A", -0.2), ("B", -1.7))),
        _token(response[second_value + 1 :], -0.01),
    ]

    _attach_domain_scores(parsed, response, token_logprobs, ("off", "on"))

    assert parsed["decisions"][0]["selected_value"] == "on"
    assert parsed["decisions"][0]["domain_scores"]["by_state"] == {
        "off": -2.0,
        "on": -0.1,
    }
    assert parsed["decisions"][0]["token_scores"] == {
        "selected_token": {"token": "B", "logprob": -0.1},
        "top_logprobs": [
            {"token": "A", "logprob": -2.0},
            {"token": "B", "logprob": -0.1},
            {"token": '"A', "logprob": -9.0},
            {"token": '"B', "logprob": -8.0},
        ],
    }
    assert parsed["decisions"][1]["selected_value"] == "off"
    assert parsed["decisions"][1]["domain_scores"]["by_state"] == {
        "off": -0.2,
        "on": -1.7,
    }


def test_missing_logprobs_leaves_response_unchanged() -> None:
    parsed = {
        "decisions": [
            {
                "selected_value": "B",
                "domain_scores": {"by_state": {"off": 0.0, "on": 0.0}},
                "token_scores": {"top_logprobs": []},
            }
        ]
    }

    _attach_domain_scores(parsed, '{"decisions":[]}', None, ("off", "on"))

    assert parsed == {"decisions": [{"selected_value": "on"}]}


def test_partial_or_invalid_logprobs_never_create_an_incomplete_vector() -> None:
    response = '{"decisions":[{"selected_value":"1"}]}'
    value_offset = response.index('"selected_value":"1"') + len('"selected_value":"')
    without_alternatives = [
        _token(response[:value_offset], -0.01),
        SimpleNamespace(token="1", bytes=[49], logprob=-0.1),
        _token(response[value_offset + 1 :], -0.01),
    ]
    parsed = {"decisions": [{"selected_value": "1"}]}

    _attach_domain_scores(parsed, response, without_alternatives, ("0", "1"))

    assert "domain_scores" not in parsed["decisions"][0]
    assert parsed["decisions"][0]["token_scores"]["top_logprobs"] == []

    invalid = [*without_alternatives]
    invalid[1] = SimpleNamespace(token="1", bytes=[49], logprob=None)
    untouched = {"decisions": [{"selected_value": "1"}]}

    _attach_domain_scores(untouched, response, invalid, ("0", "1"))

    assert untouched == {"decisions": [{"selected_value": "1"}]}


def test_bucket_prompt_uses_short_codes_for_multitoken_states() -> None:
    network = BayesianNetwork.from_structure(
        variable_states={"A": ("Treatment_success", "Treatment_failure")},
        children={},
    )
    bucket = BucketSpec(
        variable="A",
        is_evidence=False,
        observed_value=None,
        local_scope=("A",),
        separator=(),
        context_rows=[{}],
        incoming_messages=[],
    )

    prompt = build_bucket_prompt(bucket, network, {}, None, {})

    assert build_state_code_map(network.variables["A"].states) == {
        "A": "Treatment_success",
        "B": "Treatment_failure",
    }
    assert '"candidate_state_codes": {' in prompt
    assert '"A": "Treatment_success"' in prompt
    assert '"selected_value": "A | B"' in prompt


def test_raw_token_scores_reach_compiled_message_row() -> None:
    network = BayesianNetwork.from_structure(
        variable_states={"A": ("off", "on")}, children={}
    )
    bucket = BucketSpec(
        variable="A",
        is_evidence=False,
        observed_value=None,
        local_scope=("A",),
        separator=(),
        context_rows=[{}],
        incoming_messages=[],
    )
    token_scores = DecisionTokenScores(
        selected_token=TokenScore(token="B", logprob=-0.1),
        top_logprobs=[
            TokenScore(token="B", logprob=-0.1),
            TokenScore(token="A", logprob=-2.0),
        ],
    )
    response = BucketResponse.model_validate(
        {
            "variable": "A",
            "decisions": [
                {
                    "context": {},
                    "selected_value": "on",
                    "confidence": "high",
                    "rationale": "test",
                    "token_scores": token_scores.model_dump(),
                }
            ],
        }
    )

    message = semantic_message_from_response(
        response, bucket, network, build_alias_map(network, {}), {}
    )

    assert message.rows[0].token_scores == token_scores


def test_client_retries_without_logprobs_when_model_rejects_them() -> None:
    class UnsupportedLogprobsError(Exception):
        status_code = 400

    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise UnsupportedLogprobsError("model does not support logprobs")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"variable":"A","decisions":[{"context":{},'
                            '"selected_value":"1","confidence":"high",'
                            '"rationale":"x"}]}'
                        )
                    ),
                    logprobs=None,
                )
            ]
        )

    client = LLMJsonClient.__new__(LLMJsonClient)
    client.config = SimpleNamespace(
        local_model="test-model",
        llm_max_retries=0,
        show_llm_prompt=False,
        show_llm_output=False,
        openai_temperature=0.0,
        openai_use_json_response_format=False,
    )
    client.use_real_llm = False
    client.dry_run = False
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    response, _ = client.complete_json(
        purpose="bucket_argmax",
        variable="A",
        prompt="prompt",
        response_model=BucketResponse,
        candidate_states=("0", "1"),
    )

    assert response.decisions[0].domain_scores is None
    assert calls[0]["logprobs"] is True
    assert calls[0]["top_logprobs"] == 20
    assert "logprobs" not in calls[1]

    client.complete_json(
        purpose="bucket_argmax",
        variable="A",
        prompt="prompt",
        response_model=BucketResponse,
        candidate_states=("0", "1"),
    )

    assert len(calls) == 3
    assert "logprobs" not in calls[2]


def test_complete_domain_scores_survive_pickle_round_trip() -> None:
    row = MessageRow(
        context={"B": "on"},
        selected_value="Treatment_success",
        rationale="test",
        domain_scores=DomainScores(
            by_state={"Treatment_success": -0.1, "Treatment_failure": -2.0}
        ),
        token_scores=DecisionTokenScores(
            selected_token=TokenScore(token="A", logprob=-0.1),
            top_logprobs=[
                TokenScore(token="A", logprob=-0.1),
                TokenScore(token="B", logprob=-2.0),
                TokenScore(token='"A', logprob=-3.0),
            ],
        ),
    )

    restored = pickle.loads(pickle.dumps(row))

    assert restored.domain_scores == row.domain_scores
    assert restored.token_scores == row.token_scores
