from core.response_pipeline import PipelineStage, ResponsePipeline


class _MissingModelStage(PipelineStage):
    name = "generate"

    def run(self, ctx):
        raise RuntimeError("model 'missing-model' not found (status code: 404)")


def test_pipeline_reports_generation_provider_error_instead_of_generic_stuck_text():
    pipeline = ResponsePipeline(provider=object())
    pipeline._stages = {"generate": _MissingModelStage()}
    pipeline._profiles = {"BALANCED": ("generate",)}

    result = pipeline.run(
        route="chat",
        user_msg="hello",
        state={},
        meta={"model": "missing-model"},
        retrieved_memories=[],
        traits={},
        policies={},
    )

    assert result.status == "error"
    assert result.structured_output["status"] == "error"
    assert "Ollama cannot find model" in result.text
    assert "I got stuck during generation" not in result.text
