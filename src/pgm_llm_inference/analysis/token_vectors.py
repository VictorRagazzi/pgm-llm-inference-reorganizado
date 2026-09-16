"""Raw next-token score vectors from compiled semantic decisions."""

from __future__ import annotations

import json
import math

import pandas as pd

from pgm_llm_inference.mpe.compile import CompiledSemanticMessages


def compiled_token_vector_table(compiled: CompiledSemanticMessages) -> pd.DataFrame:
    """Return one record per available raw token alternative and context row.

    The generated token is included separately when the provider omits it
    from ``top_logprobs``. Repeated token strings remain separate records.
    """
    records: list[dict[str, object]] = []
    for variable, message in compiled.messages.items():
        for row_index, row in enumerate(message.rows):
            scores = getattr(row, "token_scores", None)
            if scores is None:
                continue
            generated = scores.selected_token
            alternatives = [
                (item.token, item.logprob, "top_k")
                for item in scores.top_logprobs
            ]
            if not any(
                token == generated.token
                and math.isclose(logprob, generated.logprob, abs_tol=1e-6)
                for token, logprob, _ in alternatives
            ):
                alternatives.append(
                    (generated.token, generated.logprob, "generated")
                )
            alternatives.sort(key=lambda item: item[1], reverse=True)
            context = json.dumps(row.context, ensure_ascii=False, sort_keys=True)
            for rank, (token, logprob, source) in enumerate(alternatives, start=1):
                records.append(
                    {
                        "variable": variable,
                        "row_index": row_index,
                        "context": context,
                        "selected_value": row.selected_value,
                        "rank": rank,
                        "token": token,
                        "logprob": logprob,
                        "source": source,
                        "is_generated": (
                            token == generated.token
                            and math.isclose(
                                logprob, generated.logprob, abs_tol=1e-6
                            )
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)
